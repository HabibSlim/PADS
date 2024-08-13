"""
Grouping all dataloaders for shape optimization in one place.
"""

import json
import os
import numpy as np
import fast_simplification
from datasets.metadata import (
    COMPAT_CLASSES,
    int_to_hex,
)
from datasets.sampling import get_sampling_function, normalize_pc
from util.misc import CUDAMesh
from voxelize.preprocess import robust_pcu_to_manifold


"""
Utility functions.
"""


def decimate_mesh(mesh, factor):
    """
    Decimate the input mesh by the given factor using Fast Quadric Mesh Simplification.
    """
    vertices, faces = mesh.trimesh_mesh.vertices, mesh.trimesh_mesh.faces
    vertices_out, faces_out = fast_simplification.simplify(vertices, faces, factor)
    return CUDAMesh(vertices_out, faces_out)


"""
Defining the dataset classes.
"""


class SingleManifoldDataset:
    """
    Sampling from a single mesh using various strategies.
    """

    MAX_FACES = 500000
    MAX_SAMPLE_SIZE = 2**17

    def __init__(
        self,
        obj_dir,
        shape_cls,
        n_points,
        *,
        normalize=False,
        sampling_method="surface",
        contain_method="occnets",
        max_it=10000,
        near_surface_noise=0.05,
        decimate=True,
        sample_first=False,
        batch_size=1,
        split="all",
        to_cuda=True,
    ):
        self.n_points = n_points
        self.mesh_idx = 0
        self.mesh = None
        self.normalize = normalize
        self.max_it = max_it
        self.decimate = decimate
        self.sample_first = sample_first
        self.batch_size = batch_size
        self.sampling_fn = get_sampling_function(
            sampling_method, noise_std=near_surface_noise, contain_method=contain_method
        )
        self.obj_dir = obj_dir
        self.shape_cls = shape_cls
        self.split = split
        self.to_cuda = to_cuda

        self.init_class_objs()

    def get_mesh(self, idx=None):
        """
        Load the mesh from the given index.
        """
        if idx is None:
            idx = self.mesh_idx

        if self.mesh is None:
            self.mesh = CUDAMesh.load(self.obj_files[idx], to_cuda=self.to_cuda)

            # Print an alert if the mesh is not watertight
            if not self.mesh.is_watertight:
                print("Mesh is not watertight! Performing robust conversion...")
                obj_base_name = os.path.basename(self.obj_files[idx])
                robust_pcu_to_manifold(self.obj_files[idx], "/tmp/" + obj_base_name)
                # Try to load and test if watertight
                self.mesh = CUDAMesh.load("/tmp/" + obj_base_name, to_cuda=self.to_cuda)
                if not self.mesh.is_watertight:
                    raise ValueError("Watertight conversion failed!")

                # Replace the original mesh with the watertight one
                # Write to original file
                self.mesh.export(self.obj_files[idx])
                print("Watertight conversion successful!")

            # Decimate the mesh if it has too many faces
            if self.decimate and len(self.mesh.faces) > self.MAX_FACES:
                # The ratio is the percentage of faces to REMOVE
                ratio = 1 - self.MAX_FACES / len(self.mesh.faces)
                self.mesh = decimate_mesh(self.mesh, ratio)

        return self.mesh

    def __len__(self):
        return len(self.obj_files)

    def __getitem__(self, idx):
        if self.mesh_idx != idx or self.mesh is None:
            self.mesh_idx = idx
            self.get_mesh(idx)

        # Optionally: first sample n_points first
        # And simply serve the same points for the rest of the iterations
        if self.sample_first:
            # Use batch sampling
            n_batches = self.n_points // self.MAX_SAMPLE_SIZE
            all_points, all_occs = [], []
            for k in range(n_batches):
                if k % 4 == 0:
                    print("Sampling batch [%d/%d]" % (k + 1, n_batches))
                points, occs = self.sampling_fn(self.mesh, self.MAX_SAMPLE_SIZE)
                all_points += [points]
                all_occs += [occs]
            print()
            points_idx = list(range(len(all_points)))

        # Resample the mesh
        for _ in range(self.max_it):
            if self.sample_first:
                rnd_idx = np.random.choice(points_idx)
                points = all_points[rnd_idx]
                occs = all_occs[rnd_idx]
            else:
                points, occs = self.sampling_fn(self.mesh, self.n_points)

            # Optionally: normalize the point cloud
            if self.normalize:
                points = normalize_pc(points)
            yield points, occs


class CoMPaTManifoldDataset(SingleManifoldDataset):
    """
    Sampling from a 3DCoMPaT manifold mesh dataset.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.normalize:
            print(
                "normalize=True but 3DCoMPaT shapes are already normalized to their bounding boxes."
            )

    def init_class_objs(self):
        """
        Set the list of objects for a given class/split.
        """

        def join_all(in_dir, files):
            return [os.path.join(in_dir, f) for f in files]

        compat_cls_code = int_to_hex(COMPAT_CLASSES[self.shape_cls])
        obj_files = os.listdir(self.obj_dir)
        # obj_files = [os.path.join(self.obj_dir, f) for f in obj_files]
        obj_files = [
            f for f in obj_files if f.endswith(".obj") and compat_cls_code + "_" in f
        ]
        obj_files = sorted(obj_files)

        if self.split == "all":
            self.obj_files = join_all(self.obj_dir, obj_files)
            return

        # Open the split metadata
        pwd = os.path.dirname(os.path.realpath(__file__))
        split_dict = json.load(open(os.path.join(pwd, "CoMPaT", "split.json")))

        # Filter split meshes
        obj_files = [f for f in obj_files if f.split(".")[0] in split_dict[self.split]]

        self.obj_files = join_all(self.obj_dir, obj_files)


class PartNetManifoldDataset(SingleManifoldDataset):
    """
    Sampling from a PartNet manifold mesh dataset.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def init_class_objs(self):
        """
        Get the list of objects for a given class/split.
        """

        def join_all(in_dir, files):
            return sorted([os.path.join(in_dir, f) for f in files])

        # List of fully or almost-fully segmented shapes
        pwd = os.path.dirname(os.path.realpath(__file__))
        full_segment_shapes = json.load(
            open(os.path.join(pwd, "PartNet", "full_segment_shapes.json"))
        )
        full_segment_shapes = set(full_segment_shapes)

        obj_files = os.listdir(self.obj_dir)
        obj_files = [f for f in obj_files if f.split(".")[0] in full_segment_shapes]
        obj_files = [f for f in obj_files if f.endswith(".obj")]

        if self.shape_cls == "all":
            self.obj_files = join_all(self.obj_dir, obj_files)
            return

        # Open the split metadata
        model_map = json.load(open(os.path.join(pwd, "PartNet", "shape_classes.json")))

        # Filter split meshes
        avail_models = set([os.path.basename(f).split(".")[0] for f in obj_files])
        class_models = set(model_map[self.shape_cls])
        obj_files = [
            os.path.join(self.obj_dir, model_id + ".obj")
            for model_id in class_models & avail_models
        ]

        self.obj_files = join_all(self.obj_dir, obj_files)
