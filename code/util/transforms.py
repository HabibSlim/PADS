"""
Defining random transformation matrices for data augmentation.
"""

import numpy as np


def random_uniform_scaling(min_scale=0.5, max_scale=2.0):
    """
    Generate a random uniform scaling matrix.
    """
    scale = np.random.uniform(min_scale, max_scale)
    return np.diag([scale, scale, scale, 1])


def random_scaling(min_scale=(0.5, 0.5, 0.5), max_scale=(2.0, 2.0, 2.0)):
    """
    Generate a random scaling matrix with independent scaling factors for each axis.
    """
    sx = np.random.uniform(min_scale[0], max_scale[0])
    sy = np.random.uniform(min_scale[1], max_scale[1])
    sz = np.random.uniform(min_scale[2], max_scale[2])
    return np.diag([sx, sy, sz, 1])


def random_rotation(max_angle_x=np.pi, max_angle_y=np.pi, max_angle_z=np.pi):
    """
    Generate a random rotation matrix using Euler angles.
    """

    def rotation_matrix(axis, theta):
        """
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
        """
        axis = np.asarray(axis)
        axis = axis / np.sqrt(np.dot(axis, axis))
        a = np.cos(theta / 2.0)
        b, c, d = -axis * np.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array(
            [
                [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac), 0],
                [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab), 0],
                [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc, 0],
                [0, 0, 0, 1],
            ]
        )

    # Generate random rotation angles
    rx = np.random.uniform(-max_angle_x, max_angle_x)
    ry = np.random.uniform(-max_angle_y, max_angle_y)
    rz = np.random.uniform(-max_angle_z, max_angle_z)

    # Compute rotation matrices for each axis
    Rx = rotation_matrix([1, 0, 0], rx)
    Ry = rotation_matrix([0, 1, 0], ry)
    Rz = rotation_matrix([0, 0, 1], rz)

    # Combine rotations
    return np.dot(Rz, np.dot(Ry, Rx))


def random_translation(min_trans=(-1.0, -1.0, -1.0), max_trans=(1.0, 1.0, 1.0)):
    """
    Generate a random translation matrix.
    """
    tx = np.random.uniform(min_trans[0], max_trans[0])
    ty = np.random.uniform(min_trans[1], max_trans[1])
    tz = np.random.uniform(min_trans[2], max_trans[2])
    return np.array([[1, 0, 0, tx], [0, 1, 0, ty], [0, 0, 1, tz], [0, 0, 0, 1]])


def random_transformation_matrix(
    min_scale=(0.5, 0.5, 0.5),
    max_scale=(2.0, 2.0, 2.0),
    min_trans=(-1.0, -1.0, -1.0),
    max_trans=(1.0, 1.0, 1.0),
    max_angle_x=np.pi,
    max_angle_y=np.pi,
    max_angle_z=np.pi,
    uniform_scaling=False,
):
    """
    Generate a random 3D transformation matrix combining scaling, translation, and rotation.
    """
    if uniform_scaling:
        S = random_uniform_scaling(min(min_scale), max(max_scale))
    else:
        S = random_scaling(min_scale, max_scale)
    T = random_translation(min_trans, max_trans)
    R = random_rotation(max_angle_x, max_angle_y, max_angle_z)

    # Combine transformations: first scale, then rotate, then translate
    return np.dot(T, np.dot(R, S))
