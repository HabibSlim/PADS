"""
Utility functions to prepare voxelized meshes for training.
"""

import contextlib
import sys
import os
import torch
import trimesh
import numpy as np
import point_cloud_utils as pcu
from scipy.ndimage import zoom
from .flood_fill.fill_voxels import fill_inside_voxels_cpu

BIN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin")


@contextlib.contextmanager
def stdout_redirected(to=os.devnull):
    """
    Redirect standard output (by default: to os.devnull).
    Useful to silence the noisy output of Blender.
    """
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(to, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different


@contextlib.contextmanager
def stderr_redirected(to=os.devnull):
    """
    Redirect standard error (by default: to os.devnull).
    Useful to silence the noisy output of Blender.
    """
    fd = sys.stderr.fileno()

    def _redirect_stderr(to):
        sys.stderr.close()
        os.dup2(to.fileno(), fd)
        sys.stderr = os.fdopen(fd, "w")

    with os.fdopen(os.dup(fd), "w") as old_stderr:
        with open(to, "w") as file:
            _redirect_stderr(to=file)
        try:
            yield
        finally:
            _redirect_stderr(to=old_stderr)


def get_extent(test_vox):
    test_vox = test_vox.squeeze()
    return np.max(np.argwhere(test_vox > 0.5), axis=0) - np.min(
        np.argwhere(test_vox > 0.5), axis=0
    )


def voxelize_obj(
    obj_file,
    grid_size,
    full_grid_size=64,
    vox_method="binvox",
    edge_factor=2.0,
    max_iter=100,
    per_cell=None,
):
    """
    Voxelize a manifold mesh using the specified method.
    """
    mesh = trimesh.load(obj_file)
    max_extent = np.max(mesh.bounding_box.extents)
    mesh.apply_scale(1.0 / max_extent)

    # Capture all outputs from following call
    if vox_method == "binvox":
        mesh_voxelized = mesh.voxelized(
            dimension=grid_size,
            pitch=1 / grid_size,
            method=vox_method,
            binvox_path=os.path.join(BIN_PATH, "binvox"),
        )
    elif vox_method == "ray":
        mesh_voxelized = mesh.voxelized(
            pitch=1 / grid_size, method=vox_method, per_cell=per_cell
        )
    elif vox_method == "subdivide":
        mesh_voxelized = mesh.voxelized(
            pitch=1 / grid_size,
            method=vox_method,
            edge_factor=edge_factor,
            max_iter=max_iter,
        )

    # Place the small voxel grid at 0,0,0 in the full grid
    # do NOT center the voxels
    full_grid = np.zeros((full_grid_size, full_grid_size, full_grid_size))
    if vox_method == "binvox":
        full_grid[
            int((full_grid_size - grid_size) / 2) : int(
                (full_grid_size + grid_size) / 2
            ),
            int((full_grid_size - grid_size) / 2) : int(
                (full_grid_size + grid_size) / 2
            ),
            int((full_grid_size - grid_size) / 2) : int(
                (full_grid_size + grid_size) / 2
            ),
        ] = mesh_voxelized.matrix
    else:
        grid_sizes = np.array(mesh_voxelized.matrix.shape)
        full_grid[
            int((full_grid_size - grid_sizes[0]) / 2) : int(
                (full_grid_size + grid_sizes[0]) / 2
            ),
            int((full_grid_size - grid_sizes[1]) / 2) : int(
                (full_grid_size + grid_sizes[1]) / 2
            ),
            int((full_grid_size - grid_sizes[2]) / 2) : int(
                (full_grid_size + grid_sizes[2]) / 2
            ),
        ] = mesh_voxelized.matrix
    return full_grid


def resize_voxels(
    original_voxels, subset_size=None, scaling_factor=None, mode="nearest"
):
    """
    Resize the input voxels to the desired size.
    """
    # Defining the mean statistics for the ShapeNet dataset
    mean_stats_swalk = np.array([48.024, 59.27, 46.402])
    mean_stats_real = np.array([31.0, 44.6, 29.2])
    mean_stats_ratio = mean_stats_swalk / mean_stats_real

    original_voxels = original_voxels.squeeze()
    orig_size = original_voxels.shape
    # compute subset_size using mean_stats_ratio
    if subset_size is None:
        subset_size = np.round(orig_size / mean_stats_ratio).astype(int)

    # Calculate the scaling factors for each axis
    if scaling_factor is None:
        scaling_factors = (
            subset_size[0] / original_voxels.shape[0],
            subset_size[1] / original_voxels.shape[1],
            subset_size[2] / original_voxels.shape[2],
        )
    else:
        scaling_factors = (scaling_factor, scaling_factor, scaling_factor)

    # Use the zoom function to resample the voxels
    scaled_voxels = zoom(original_voxels, zoom=scaling_factors, order=0, mode=mode)

    # Replace voxels in a grid of the original size
    scaled_voxels = np.pad(
        scaled_voxels,
        (
            (0, orig_size[0] - scaled_voxels.shape[0]),
            (0, orig_size[1] - scaled_voxels.shape[1]),
            (0, orig_size[2] - scaled_voxels.shape[2]),
        ),
    )
    # expand dims
    scaled_voxels = np.expand_dims(scaled_voxels, axis=0)
    return scaled_voxels


def center_voxels(voxels):
    """
    Center the voxels by computing extent and translating.
    Voxels are assumed to be binary.
    Voxels is a 3D numpy tensor.
    """
    # Compute the center of mass of the voxels
    center = np.mean(np.argwhere(voxels), axis=0)

    # Compute the shift needed to center the voxels in the grid
    shift = np.array(voxels.shape) / 2.0 - center

    # Create an empty array of the same shape as the input voxels
    centered_voxels = np.zeros_like(voxels)

    # Iterate over all voxels
    for x, y, z in np.argwhere(voxels):
        # Compute the new coordinates of the voxel after the shift
        new_coords = np.round([x, y, z] + shift).astype(int)

        # Check if the new coordinates are within the grid
        if np.all(new_coords >= 0) and np.all(new_coords < np.array(voxels.shape)):
            # Set the voxel at the new coordinates in the centered voxels grid
            centered_voxels[tuple(new_coords)] = voxels[x, y, z]

    return centered_voxels


def mesh_to_manifold(obj_model, out_file, depth=8, extra_params=""):
    """
    Convert the input mesh to a manifold mesh.
    """
    os.system(
        "%s/manifold --input %s --output %s --depth %d %s"
        % (BIN_PATH, obj_model, out_file, depth, extra_params)
    )


def robust_pcu_to_manifold(obj_model, out_file, resolution=100_000):
    """
    Robust version of ManifoldPlus with lessened fidelity.
    """
    obj = trimesh.load_mesh(obj_model)
    v, f = obj.vertices, obj.faces
    vw, fw = pcu.make_mesh_watertight(v, f, resolution)
    new_obj = trimesh.Trimesh(vw, fw)
    new_obj.export(out_file)


def process_mesh(obj_model, out_path, out_path_vox, flood_fill=False):
    """
    Process a single mesh file.
    """
    out_obj_file = os.path.join(out_path, os.path.basename(obj_model))
    with stdout_redirected(), stderr_redirected():
        if not os.path.exists(out_obj_file):
            mesh_to_manifold(obj_model, out_obj_file)
        mesh_voxels = voxelize_obj(
            out_obj_file,
            256,
            full_grid_size=256,
            vox_method="binvox",
        )
    mesh_voxels = np.expand_dims(mesh_voxels, axis=0)
    mesh_voxels = np.flip(mesh_voxels, axis=3)

    if flood_fill:
        # Convert mesh_voxels to torch tensor
        mesh_voxels = torch.from_numpy(mesh_voxels.copy()).float()

        # Flood fill the generated voxels
        mesh_voxels = fill_inside_voxels_cpu(mesh_voxels)

    # Convert to boolean numpy array
    centered_voxs = np.array(mesh_voxels > 0.5, dtype=bool)
    centered_voxs = np.packbits(centered_voxs, axis=-1)

    # Save the voxelized mesh
    out_file_vox = os.path.join(
        out_path_vox, os.path.basename(obj_model).replace(".obj", ".npy")
    )
    np.save(out_file_vox, centered_voxs)
