"""
Gathering all the transforms in one place.
"""
import torch
import numpy as np


SCALE_FCT_RND = np.array([0.46350697, 0.35710097, 0.40755142])
SCALE_FCT_NORMAL = np.array([0.33408034, 0.39906635, 0.35794342])


@torch.inference_mode()
def center_in_unit_sphere(pc, in_place=True):
    """
    Center the point cloud in the unit sphere.
    """
    if type(pc) == torch.Tensor:
        if not in_place:
            pc = pc.clone()

        for axis in range(3):
            r_max = torch.max(pc[:, axis])
            r_min = torch.min(pc[:, axis])
            gap = (r_max + r_min) / 2.0
            pc[:, axis] -= gap

        largest_distance = torch.max(torch.sqrt(torch.sum(pc**2, dim=1)))
        pc /= largest_distance
    else:
        if not in_place:
            pc = pc.copy()

        for axis in range(3):  # center around each axis
            r_max = np.max(pc[:, axis])
            r_min = np.min(pc[:, axis])
            gap = (r_max + r_min) / 2.0
            pc[:, axis] -= gap

        largest_distance = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc /= largest_distance

    return pc


def get_pc_ae_transform(args):
    """
    Transform pointcloud for the PC-AE encoder.
    """

    def transform(x):
        if type(x) == torch.Tensor:
            x = x.cpu().numpy()
            return center_in_unit_sphere(np.matmul(rotate_matrix, x.T).T) * scale_factor
        else:
            return center_in_unit_sphere(np.matmul(rotate_matrix, x.T).T) * scale_factor

    # Defining transformation
    if "RND" in args.data_path:
        scale_factor = SCALE_FCT_RND
    else:
        scale_factor = SCALE_FCT_NORMAL

    rotate_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])

    return transform


def get_modelnet40_transform(pc):
    """
    Transform pointcloud for the ModelNet40 dataset.
    """
    pc = pc @ np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]).astype(np.float32)
    pc = pc @ np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]).astype(np.float32)
    pc = pc @ np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]).astype(np.float32)
    pc = center_in_unit_sphere(pc)

    return pc


def get_imnet_transform(args):
    """
    Transform pointcloud for the IMNet autoencoder.
    """
    # Defining the mean statistics for the ShapeNet dataset
    # mean_stats_swalk = np.array([48.024, 59.27, 46.402])
    # mean_stats_real = np.array([31.0, 44.6, 29.2])
    # mean_stats_ratio = mean_stats_real / mean_stats_swalk
    rotate_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])

    t_rec = lambda x: center_in_unit_sphere(x)
    t_mesh = lambda x: center_in_unit_sphere(x @ rotate_matrix)

    return (t_rec, t_mesh)
