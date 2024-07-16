"""
Grouping all dataloaders for shape optimization in one place.
"""

import os
import numpy as np
import torch
import fast_simplification
from functools import partial
from datasets.metadata import COMPAT_CLASSES, int_to_hex
from datasets.sampling import (
    sample_surface_simple,
    sample_near_surface,
    sample_volume,
    combine_samplings,
)
from util.misc import CUDAMesh
from voxelize.preprocess import robust_pcu_to_manifold


"""
Utility functions.
"""


def get_class_objs(obj_dir, shape_cls):
    """
    Get the list of objects for a given class.
    """
    compat_cls_code = int_to_hex(COMPAT_CLASSES[shape_cls])
    obj_files = os.listdir(obj_dir)
    obj_files = [os.path.join(obj_dir, f) for f in obj_files]
    obj_files = [
        f for f in obj_files if f.endswith(".obj") and compat_cls_code + "_" in f
    ]
    obj_files = sorted(obj_files)
    return obj_files


def decimate_mesh(mesh, factor):
    """
    Decimate the input mesh by the given factor using Fast Quadric Mesh Simplification.
    """
    vertices, faces = mesh.vertices, mesh.faces
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
        noise_std=0.05,
        decimate=True,
        sample_first=False,
        batch_size=1,
    ):
        self.n_points = n_points
        self.obj_files = get_class_objs(obj_dir, shape_cls)
        self.mesh_idx = 0
        self.mesh = None
        self.normalize = normalize
        self.max_it = max_it
        self.decimate = decimate
        self.sample_first = sample_first
        self.batch_size = batch_size

        # Defining sampling strategies
        fn_sample_surface = partial(sample_surface_simple)
        fn_sample_near_surface = partial(
            sample_near_surface,
            noise_std=noise_std,
            contain_method=contain_method,
        )
        fn_sample_volume = partial(sample_volume, contain_method=contain_method)

        self.sampling_fn = {
            "surface": fn_sample_surface,
            "near_surface": fn_sample_near_surface,
            "volume": fn_sample_volume,
            "volume+surface": partial(
                combine_samplings,
                sampling_fns=[
                    fn_sample_volume,
                    fn_sample_surface,
                ],
            ),
            "volume+near_surface": partial(
                combine_samplings,
                sampling_fns=[fn_sample_volume, fn_sample_near_surface],
            ),
        }[sampling_method]

    def __len__(self):
        return len(self.obj_files)

    def __getitem__(self, idx):
        if self.mesh is None or self.mesh_idx != idx:
            self.mesh = CUDAMesh.load(self.obj_files[idx])
            self.mesh_idx = idx

            # Print an alert if the mesh is not watertight
            if not self.mesh.is_watertight:
                print("Mesh is not watertight! Performing robust conversion...")
                obj_base_name = os.path.basename(self.obj_files[idx])
                robust_pcu_to_manifold(self.obj_files[idx], "/tmp/" + obj_base_name)
                # Try to load and test if watertight
                self.mesh = CUDAMesh.load("/tmp/" + obj_base_name)
                assert self.mesh.is_watertight
                # Replace the original mesh with the watertight one
                # Write to original file
                self.mesh.export(self.obj_files[idx])
                print("Watertight conversion successful!")

            # Decimate the mesh if it has too many faces
            if self.decimate and len(self.mesh.faces) > self.MAX_FACES:
                # The ratio is the percentage of faces to REMOVE
                ratio = 1 - self.MAX_FACES / len(self.mesh.faces)
                self.mesh = decimate_mesh(self.mesh, ratio)

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
                points = self.normalize_pc(torch.tensor(points).float())
            yield points, occs
