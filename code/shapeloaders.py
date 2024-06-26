"""
Grouping all dataloaders for shape optimization in one place.
"""

import os
import trimesh
import numpy as np
import torch
import fast_simplification
from functools import partial
from util.sampling import sample_surface_tpp, sample_surface_trimesh
from voxelize.preprocess import robust_pcu_to_manifold
from metadata import COMPAT_CLASSES, int_to_hex
from util.contains.inside_mesh import is_inside


"""
Sampling utility functions.
"""


def decimate_mesh(mesh, factor):
    """
    Decimate the input mesh by the given factor using Fast Quadric Mesh Simplification.
    """
    vertices, faces = mesh.vertices, mesh.faces
    vertices_out, faces_out = fast_simplification.simplify(vertices, faces, factor)
    return trimesh.Trimesh(vertices_out, faces_out)


def safe_sample_surface(mesh, num_points, sample_method="triangle_point_picking"):
    """
    Sample points on the mesh surface. Guarantee that the number of points is num_points.
    """
    if sample_method == "triangle_point_picking":
        return sample_surface_tpp(mesh, num_points)
    elif sample_method == "trimesh":
        points = sample_surface_trimesh(mesh, num_points)
        if len(points) < num_points:
            # Resample points
            new_points = sample_surface_trimesh(mesh, num_points)
            points = np.concatenate([points, new_points[: num_points - len(points)]])
            np.random.shuffle(points)
            points = torch.tensor(points).float().cuda()
        return points


def sample_volume(
    mesh, num_points, hash_resolution=512, contain_method="occnets", get_occ=True
):
    """
    Randomly sample points in the unit cube.
    Return occupancies for the input mesh.
    """
    points = torch.randn(num_points, 3) * 2 - 1
    points = points.cuda()

    if get_occ:
        contains = is_inside(mesh, points, hash_resolution, contain_method)
        return points, contains
    return points


def sample_surface_simple(mesh, num_points, get_occ=True):
    """
    Randomly sample points on the mesh surface.
    """
    points = safe_sample_surface(mesh, num_points)
    if get_occ:
        return points, torch.ones(num_points)
    return points


def sample_near_surface(
    mesh,
    num_points,
    get_occ=True,
    noise_std=0.05,
    hash_resolution=512,
    contain_method="occnets",
):
    """
    Randomly sample points close to the mesh surface.
    """
    points = safe_sample_surface(mesh, num_points)
    # Add noise to the points
    points += torch.randn(num_points, 3).cuda() * noise_std
    contains = is_inside(mesh, points, hash_resolution, contain_method)

    return points, contains


def combine_samplings(mesh, num_points, sampling_fns):
    """
    Sample points using multiple sampling functions.
    """
    points, occs = [], []
    for sampling_fn in sampling_fns:
        p, o = sampling_fn(mesh, num_points // len(sampling_fns), get_occ=True)
        points.append(p)
        occs.append(o)
    return torch.cat(points), torch.cat(occs)


def normalize_pc(point_cloud, use_center_of_bounding_box=True):
    """
    Normalize the point cloud to be in the range [-1, 1] and centered at the origin.
    """
    min_x, max_x = torch.min(point_cloud[:, 0]), torch.max(point_cloud[:, 0])
    min_y, max_y = torch.min(point_cloud[:, 1]), torch.max(point_cloud[:, 1])
    min_z, max_z = torch.min(point_cloud[:, 2]), torch.max(point_cloud[:, 2])
    # center the point cloud
    if use_center_of_bounding_box:
        center = torch.tensor(
            [(min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2]
        )
    else:
        center = torch.mean(point_cloud, dim=0)
    point_cloud = point_cloud - center.to(point_cloud.device)
    dist = torch.max(torch.sqrt(torch.sum((point_cloud**2), dim=1)))
    point_cloud = point_cloud / dist  # scale the point cloud
    return point_cloud * 8.0


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


"""
Defining the dataset classes.
"""


class SingleManifoldDataset:
    """
    Sampling from a single mesh using various strategies.
    """

    MAX_FACES = 500000
    SAMPLE_BATCH_SIZE = 2**17

    def __init__(
        self,
        obj_dir,
        shape_cls,
        n_points,
        normalize=False,
        sampling_method="surface",
        contain_method="occnets",
        max_it=10000,
        noise_std=0.05,
        decimate=True,
        sample_first=False,
    ):
        self.n_points = n_points
        self.obj_files = get_class_objs(obj_dir, shape_cls)
        self.obj_idx = 0
        self.obj = None
        self.normalize = normalize
        self.max_it = max_it
        self.decimate = decimate
        self.sample_first = sample_first

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
        if self.obj is None or self.obj_idx != idx:
            self.obj = trimesh.load(self.obj_files[idx])
            self.obj_idx = idx

            # Print an alert if the mesh is not watertight
            if not self.obj.is_watertight:
                print("Mesh is not watertight! Performing robust conversion...")
                obj_base_name = os.path.basename(self.obj_files[idx])
                robust_pcu_to_manifold(self.obj_files[idx], "/tmp/" + obj_base_name)
                # Try to load and test if watertight
                self.obj = trimesh.load("/tmp/" + obj_base_name)
                assert self.obj.is_watertight
                # Replace the original mesh with the watertight one
                # Write to original file
                self.obj.export(self.obj_files[idx])
                print("Watertight conversion successful!")

            # Decimate the mesh if it has too many faces
            if self.decimate and len(self.obj.faces) > self.MAX_FACES:
                # The ratio is the percentage of faces to REMOVE
                ratio = 1 - self.MAX_FACES / len(self.obj.faces)
                self.obj = decimate_mesh(self.obj, ratio)

        # Optionally: first sample n_points first
        # And simply serve the same points for the rest of the iterations
        if self.sample_first:
            # Use batch sampling
            n_batches = self.n_points // self.SAMPLE_BATCH_SIZE
            all_points, all_occs = [], []
            for k in range(n_batches):
                if k % 4 == 0:
                    print("Sampling batch [%d/%d]" % (k + 1, n_batches))
                points, occs = self.sampling_fn(self.obj, self.SAMPLE_BATCH_SIZE)
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
                points, occs = self.sampling_fn(self.obj, self.n_points)

            # Optionally: normalize the point cloud
            if self.normalize:
                points = self.normalize_pc(torch.tensor(points).float())
            yield points, occs
