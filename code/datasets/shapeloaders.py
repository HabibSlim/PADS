"""
Grouping all dataloaders for shape optimization in one place.
"""

import json
import os
import numpy as np
import torch
from torch.utils import data
import trimesh
from collections import defaultdict
from datasets.metadata import (
    class_to_hex,
    COMPAT_TRANSFORMS,
    SHAPENET_CLASSES,
)
from datasets.sampling import get_sampling_function
from util.mesh import CUDAMesh, decimate_mesh
from util.misc import load_pickle
from util.transforms import random_transformation_matrix
from voxelize.preprocess import robust_pcu_to_manifold, robust_pcu_to_manifold_in_memory


"""
Defining the dataset classes.
"""


class SingleManifoldDataset:
    """
    Sampling from a single mesh using various strategies.
    """

    MAX_FACES = 800000
    MAX_SAMPLE_SIZE = 2**17

    def __init__(
        self,
        obj_dir,
        shape_cls,
        n_points,
        *,
        sampling_method="surface",
        contain_method="occnets",
        max_it=10000,
        near_surface_noise=0.05,
        decimate=True,
        sample_first=False,
        batch_size=1,
        split="all",
        recenter_mesh=False,
        process_mesh=False,
        sampling_fn=None,
    ):
        self.n_points = n_points
        self.mesh_idx = 0
        self.mesh = None
        self.max_it = max_it
        self.decimate = decimate
        self.sample_first = sample_first
        self.batch_size = batch_size
        if sampling_fn is not None:
            self.sampling_fn = sampling_fn
        else:
            self.sampling_fn = get_sampling_function(
                sampling_method,
                noise_std=near_surface_noise,
                contain_method=contain_method,
            )
        self.obj_dir = obj_dir
        self.shape_cls = shape_cls
        self.split = split
        self.recenter_mesh = recenter_mesh
        self.process_mesh = process_mesh
        self.robust_watertight = False
        self.recenter_matrix, self.scale_matrix, self.align_matrix = None, None, None
        self.init_class_objs()

    def opt_decimate(self, mesh):
        """
        Optionally the input mesh by the given factor using Fast Quadric Mesh Simplification.
        """
        if self.decimate and len(mesh.faces) > self.MAX_FACES:
            # The ratio is the percentage of faces to REMOVE
            ratio = 1 - self.MAX_FACES / len(mesh.faces)
            return decimate_mesh(mesh, ratio)
        return mesh

    def get_mesh(self, idx=None):
        """
        Load the mesh from the given index.
        """
        if idx is None:
            idx = self.mesh_idx
        else:
            self.mesh_idx = idx

        if self.mesh_idx != idx or self.mesh is None:
            self.mesh = CUDAMesh.load(
                self.obj_files[idx],
            )

            # Print an alert if the mesh is not watertight
            if not self.mesh.is_watertight and self.robust_watertight:
                print("Mesh is not watertight! Performing robust conversion...")
                obj_base_name = os.path.basename(self.obj_files[idx])
                robust_pcu_to_manifold(self.obj_files[idx], "/tmp/" + obj_base_name)
                # Try to load and test if watertight
                self.mesh = CUDAMesh.load("/tmp/" + obj_base_name)
                if not self.mesh.is_watertight:
                    raise ValueError("Watertight conversion failed!")

                # Replace the original mesh with the watertight one
                # Write to original file
                self.mesh.export(self.obj_files[idx])
                print("Watertight conversion successful!")

            # Decimate the mesh if it has too many faces
            self.mesh = self.opt_decimate(self.mesh)

            if self.recenter_mesh:
                self.mesh, self.recenter_matrix = self.mesh.recenter()

            if self.process_mesh:
                trimesh_mesh = self.mesh.trimesh_mesh
                # Keep only non-degenerate faces
                # trimesh_mesh.update_faces(trimesh_mesh.nondegenerate_faces())
                trimesh_mesh = trimesh_mesh.process(validate=True)
                self.mesh = CUDAMesh.from_trimesh(trimesh_mesh)

        return self.mesh

    def get_model_id(self, obj_k):
        """
        Get the model id of the currently loaded mesh.
        """
        return os.path.basename(self.obj_files[obj_k]).split(".")[0]

    def __len__(self):
        return len(self.obj_files)

    def __getitem__(self, idx, force_refresh=False):
        if force_refresh or self.mesh_idx != idx or self.mesh is None:
            self.get_mesh(idx)
            self.mesh_idx = idx

        # Optionally: first sample n_points first
        # And simply serve the same points for the rest of the iterations
        if self.sample_first:
            # Use batch sampling
            n_batches = self.n_points // self.MAX_SAMPLE_SIZE
            all_points, all_occs = [], []
            for k in range(n_batches):
                points, occs = self.sampling_fn(self.mesh, self.MAX_SAMPLE_SIZE)
                all_points += [points]
                all_occs += [occs]
            points_idx = list(range(len(all_points)))

        # Resample the mesh
        for _ in range(self.max_it):
            if self.sample_first:
                rnd_idx = np.random.choice(points_idx)
                points = all_points[rnd_idx]
                occs = all_occs[rnd_idx]
            else:
                points, occs = self.sampling_fn(self.mesh, self.n_points)

            yield points, occs


class CoMPaTManifoldDataset(SingleManifoldDataset):
    """
    Sampling from a 3DCoMPaT manifold mesh dataset.
    """

    def __init__(
        self,
        *args,
        scale_to_shapenet=False,
        align_to_shapenet=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.scale_to_shapenet = scale_to_shapenet
        self.align_to_shapenet = align_to_shapenet
        if self.scale_to_shapenet:
            # Load the ShapeNet-3DCoMPaT size mapping
            all_extents = os.path.join(self.obj_dir, "..", "extents.pkl")
            class_extents = load_pickle(all_extents)[self.shape_cls]

            # Define the scale transformation matrix
            scale_vector = class_extents["shapenet"] / class_extents["compat"]
            sx, sy, sz = scale_vector
            self.scale_matrix = np.array(
                [[sx, 0, 0, 0], [0, sy, 0, 0], [0, 0, sz, 0], [0, 0, 0, 1]]
            )
        if self.align_to_shapenet:
            if self.shape_cls in COMPAT_TRANSFORMS:
                mat = np.array(COMPAT_TRANSFORMS[self.shape_cls])
                mat = np.pad(mat, (0, 1))
                mat[3, 3] = 1
            else:
                mat = np.eye(4)
            self.align_matrix = mat

    def init_class_objs(self):
        """
        Set the list of objects for a given class/split.
        """

        def join_all(in_dir, files):
            return [os.path.join(in_dir, f) for f in files]

        compat_cls_code = class_to_hex(self.shape_cls)
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

    def get_mesh(self, idx=None):
        """
        Load the mesh from the given index.
        """
        first_load = self.mesh is None or self.mesh_idx != idx
        self.mesh = super().get_mesh(idx)
        if first_load:
            if self.scale_to_shapenet:
                self.mesh = self.mesh.apply_transform(self.scale_matrix)
            if self.align_to_shapenet:
                self.mesh = self.mesh.apply_transform(self.align_matrix)
        return self.mesh


class CoMPaTSegmentDataset(CoMPaTManifoldDataset):
    """
    Sampling from a 3DCoMPaT manifold mesh dataset, with segmented parts as oriented bounding boxes.
    """

    def __init__(
        self,
        *args,
        random_transform=False,
        random_rotation=False,
        force_retransform=False,
        random_part_drop=False,
        n_parts_to_drop=1,
        remove_small_parts=False,
        min_part_volume=0.01**3,
        robust_watertight=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.bounding_boxes = defaultdict(list)
        self.random_transform = random_transform
        self.random_rotation = random_rotation
        self.force_retransform = force_retransform
        self.random_part_drop = random_part_drop
        self.n_parts_to_drop = n_parts_to_drop
        self.part_drop_cycle = []
        self.part_drop_idx = 0
        self.remove_small_parts = remove_small_parts
        self.min_part_volume = 0.01**3
        self.robust_watertight = True

    def init_class_objs(self):
        """
        Set the list of objects for a given class/split.
        """
        super().init_class_objs()
        self.pkl_files = [f.replace(".obj", ".pkl") for f in self.obj_files]

    def compose_transforms(self, transforms):
        """
        Compose a list of transformations.
        """
        full_transform = np.eye(4)
        for transform in transforms:
            if transform is not None:
                full_transform = np.matmul(full_transform, transform)
        return full_transform

    def filter_small_parts(self, mesh_segments, min_volume):
        """
        Hard-drop parts smaller than 1% of the unit cube
        """
        hard_remove = []
        for part_key, seg_mesh in mesh_segments.items():
            if seg_mesh.volume < min_volume:
                hard_remove.append(part_key)
        # print("Dropping parts:", hard_remove, "due to small volume.")

        for part_key in hard_remove:
            mesh_segments.pop(part_key)

        return mesh_segments

    def get_dropped_parts(self, init=False):
        """
        Get the list of parts to drop.
        """
        # Randomly drop parts
        to_drop = []
        if self.random_part_drop:
            if init:
                n_parts = len(self.original_seg_meshes)
                self.part_drop_cycle = np.random.permutation(n_parts)
                self.part_drop_idx = -1

            for _ in range(self.n_parts_to_drop):
                self.part_drop_idx = (self.part_drop_idx + 1) % len(
                    self.part_drop_cycle
                )
                to_drop.append(self.part_drop_cycle[self.part_drop_idx])

            # print("Dropping parts:", to_drop, "from cycle: ", self.part_drop_cycle)

        return to_drop

    def get_mesh(self, idx=None):
        """
        Load the mesh from the given index.
        """
        first_load = self.mesh is None or self.mesh_idx != idx

        if first_load:
            # Load the original mesh and cache it
            self.original_seg_meshes = load_pickle(self.pkl_files[idx])

            # Filter mesh segments which are too small
            if self.remove_small_parts:
                self.original_seg_meshes = self.filter_small_parts(
                    self.original_seg_meshes, self.min_part_volume
                )

            # Reset the bounding boxes
            self.mesh_idx = idx
            self.bounding_boxes[idx] = []
        elif self.force_retransform:
            self.bounding_boxes[idx] = []

        if first_load or self.force_retransform:
            # Compute mesh extent using the original mesh
            temp_mesh = trimesh.util.concatenate(
                list(self.original_seg_meshes.values())
            )
            mesh_extent = temp_mesh.bounding_box_oriented.extents
            max_scale_vector = 1.0 / mesh_extent
            max_translation_vector = (1.0 - mesh_extent) / 2

            # Compute random transformation matrix
            random_mat = None
            if self.random_transform:
                rot_angle = 0
                if self.random_rotation:
                    rot_angle = np.pi / 16
                random_mat = random_transformation_matrix(
                    min_scale=(0.5, 0.5, 0.5),
                    max_scale=max_scale_vector,
                    min_trans=-max_translation_vector,
                    max_trans=max_translation_vector,
                    max_angle_x=rot_angle,
                    max_angle_y=rot_angle,
                    max_angle_z=rot_angle,
                    uniform_scaling=False,
                )

            # Compose the transforms
            all_mats = [
                self.recenter_matrix,
                self.scale_matrix,
                self.align_matrix,
                random_mat,
            ]
            self.transform_mat = self.compose_transforms(all_mats)

            # Randomly drop parts
            to_drop = self.get_dropped_parts(init=first_load)

            # Check if the bounding box computation fails
            for k, (part_label, seg_mesh) in enumerate(
                self.original_seg_meshes.items()
            ):
                try:
                    bb = seg_mesh.bounding_box_oriented
                except TypeError:
                    to_drop.append(k)

            # Apply the transforms to the original mesh
            transformed_segments = []
            for k, (part_label, seg_mesh) in enumerate(
                self.original_seg_meshes.items()
            ):
                if k in to_drop:
                    continue
                transformed_seg_mesh = seg_mesh.copy().apply_transform(
                    self.transform_mat
                )
                transformed_segments.append(transformed_seg_mesh)
                self.bounding_boxes[idx].append(
                    (part_label, transformed_seg_mesh.bounding_box_oriented)
                )

            trimesh_mesh = trimesh.util.concatenate(transformed_segments)

            # Force watertight conversion
            print(
                "Watertight needed: [%s]. \n" % "yes"
                if not trimesh_mesh.is_watertight
                else "no"
            )
            if not trimesh_mesh.is_watertight and self.robust_watertight:
                print("Mesh is not watertight! Performing robust conversion...")
                trimesh_mesh = robust_pcu_to_manifold_in_memory(
                    trimesh_mesh, resolution=50_000
                )

                if not trimesh_mesh.is_watertight:
                    print(
                        "Watertight conversion failed for [%s]!" % self.obj_files[idx]
                    )

            # Create the CUDA mesh from the transformed segments
            self.mesh = CUDAMesh.from_trimesh(trimesh_mesh)
            self.mesh = self.opt_decimate(self.mesh)

        return self.mesh

    def __getitem__(self, idx):
        parent_gen = super().__getitem__(idx, force_refresh=self.force_retransform)

        # Yield the results from the parent along with bounding boxes
        for points, occs in parent_gen:
            yield points, occs, self.bounding_boxes[idx]


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


class ShapeNetDataset(data.Dataset):
    """
    Sampling from a ShapeNet dataset.
    """

    def __init__(
        self,
        dataset_folder,
        shape_cls=None,
        transform=None,
        sampling=True,
        num_samples=4096,
        pc_size=2048,
    ):
        self.pc_size = pc_size

        self.transform = transform
        self.num_samples = num_samples
        self.sampling = sampling

        self.return_surface = True
        self.surface_sampling = True

        self.point_folder = dataset_folder
        self.dataset_folder = os.path.join(dataset_folder, shape_cls)

        self.models = []
        subpath = os.path.join(self.dataset_folder, "4_pointcloud")

        self.models += [
            {"category": shape_cls, "model": m.replace(".npz", "")}
            for m in os.listdir(subpath)
        ]

    def __getitem__(self, idx):
        idx = idx % len(self.models)

        category = self.models[idx]["category"]
        model = self.models[idx]["model"]

        pc_path = os.path.join(self.dataset_folder, "4_pointcloud", model + ".npz")
        with np.load(pc_path) as data:
            surface = data["points"].astype(np.float32)
        if self.surface_sampling:
            ind = np.random.default_rng().choice(
                surface.shape[0], self.pc_size, replace=False
            )
            surface = surface[ind]
        surface = torch.from_numpy(surface).unsqueeze(0)

        return surface, SHAPENET_CLASSES[category]

    def __len__(self):
        return len(self.models)
