"""
Mesh sampling utilities.
"""

import numpy as np
import torch
import trimesh
from functools import partial
from util.contains.inside_mesh import is_inside


@torch.inference_mode()
def face_areas_normals(vertices, faces):
    """
    Get face areas and normals for a batch of meshes.
    """
    face_normals = torch.cross(
        vertices[:, faces[:, 1], :] - vertices[:, faces[:, 0], :],
        vertices[:, faces[:, 2], :] - vertices[:, faces[:, 1], :],
        dim=2,
    )
    face_areas = torch.norm(face_normals, dim=2)
    face_normals = face_normals / face_areas[:, :, None]
    face_areas = 0.5 * face_areas
    return face_areas, face_normals


@torch.inference_mode()
def sample_surface_tpp(mesh, num_points, face_dist=None):
    """
    Sample points on the surface of a mesh using Triangle Point Picking.
    CUDA-compatible sampling method.

    sample method:
    http://mathworld.wolfram.com/TrianglePointPicking.html
    """
    vertices, faces = mesh.vertices, mesh.faces
    N = vertices.shape[0]

    if face_dist is None:
        face_dist = mesh.face_distribution
    face_index = face_dist.sample((num_points,))

    # pull triangles into the form of an origin + 2 vectors
    tri_origins = vertices[:, faces[:, 0], :]
    tri_vectors = vertices[:, faces[:, 1:], :].clone()
    tri_vectors -= tri_origins.repeat(1, 1, 2).reshape((N, len(faces), 2, 3))

    # pull the vectors for the faces we are going to sample from
    face_index = face_index.transpose(0, 1)
    face_index = face_index[:, :, None].expand((N, num_points, 3))
    tri_origins = torch.gather(tri_origins, dim=1, index=face_index)
    face_index2 = face_index[:, :, None, :].expand((N, num_points, 2, 3))
    tri_vectors = torch.gather(tri_vectors, dim=1, index=face_index2)

    # randomly generate two 0-1 scalar components to multiply edge vectors by
    random_lengths = torch.rand(
        num_points, 2, 1, device=vertices.device, dtype=tri_vectors.dtype
    )

    # points will be distributed on a quadrilateral if we use 2x [0-1] samples
    # if the two scalar components sum less than 1.0 the point will be
    # inside the triangle, so we find vectors longer than 1.0 and
    # transform them to be inside the triangle
    random_test = random_lengths.sum(dim=1).reshape(-1) > 1.0
    random_lengths[random_test] -= 1.0
    random_lengths = torch.abs(random_lengths)

    # multiply triangle edge vectors by the random lengths and sum
    sample_vector = (tri_vectors * random_lengths[None, :]).sum(dim=2)

    # finally, offset by the origin to generate
    # (n,3) points in space on the triangle
    samples = sample_vector + tri_origins

    return samples


def sample_surface_trimesh(mesh, num_points):
    """
    Sample mesh surface evenly using trimesh.
    """
    pc_sampled, _ = trimesh.sample.sample_surface_even(mesh, num_points)
    return pc_sampled


def normalize_point_cloud(point_cloud, use_center_of_bounding_box=True):
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
    point_cloud = point_cloud - center
    dist = torch.max(torch.sqrt(torch.sum((point_cloud**2), dim=1)))
    point_cloud = point_cloud / dist  # scale the point cloud
    return point_cloud


"""
Sampling utility functions.
"""


def sample_surface(mesh, num_points, sample_method="triangle_point_picking"):
    """
    Sample points on the mesh surface. Guarantee that the number of points is num_points.
    """
    assert sample_method in [
        "triangle_point_picking",
        "trimesh",
    ], "Invalid sampling method"
    if sample_method == "triangle_point_picking":
        return sample_surface_tpp(mesh, num_points).detach()
    elif sample_method == "trimesh":
        points = sample_surface_trimesh(mesh, num_points)
        if len(points) < num_points:
            # Resample points
            new_points = sample_surface_trimesh(mesh, num_points)
            points = np.concatenate([points, new_points[: num_points - len(points)]])
            np.random.shuffle(points)
            points = torch.tensor(points).float().cuda()
        return points.unsqueeze(0)


def sample_cube(num_points):
    """
    Sample points in the unit cube.
    """
    points = torch.randn((1, num_points, 3))
    points = points.cuda()
    return points


def sample_distribution(
    gt_mesh,
    num_points,
    *,
    face_dist=None,
    noise_std=0.05,
    hash_resolution=512,
    contain_method="occnets",
    get_occ=True,
):
    """
    Sample faces from an input weighted face-wise distribution.
    """
    points = sample_surface_tpp(gt_mesh, num_points, face_dist=face_dist).detach()

    # Add noise to the points
    points += torch.randn(num_points, 3).cuda() * noise_std

    if get_occ:
        occs = is_inside(gt_mesh, points, hash_resolution, contain_method)
        return points, occs
    return points


def sample_volume(
    mesh, num_points, hash_resolution=512, contain_method="occnets", get_occ=True
):
    """
    Randomly sample points in the unit cube.
    Return occupancies for the input mesh.
    """
    points = sample_cube(num_points)
    if get_occ:
        occs = is_inside(mesh, points, hash_resolution, contain_method)
        return points, occs
    return points


def sample_surface_simple(mesh, num_points, get_occ=True):
    """
    Randomly sample points on the mesh surface.
    """
    points = sample_surface(mesh, num_points)
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
    points = sample_surface(mesh, num_points)
    # Add noise to the points
    points += torch.randn(num_points, 3).cuda() * noise_std
    occs = is_inside(mesh, points, hash_resolution, contain_method)

    return points, occs


def combine_samplings(mesh, num_points, sampling_fns):
    """
    Sample points using multiple sampling functions.
    """
    points, occs = [], []

    # # Distribute the points evenly among the sampling functions
    # points_per_fn = [num_points // len(sampling_fns)] * len(sampling_fns)
    # points_per_fn[-1] += num_points - sum(points_per_fn)

    for k, sampling_fn in enumerate(sampling_fns):
        p, o = sampling_fn(mesh, num_points // len(sampling_fns), get_occ=True)
        points.append(p)
        occs.append(o)
    return torch.cat(points), torch.cat(occs)
    # torch.cat(points, dim=1), torch.cat(occs)


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


"""
Generating custom sampling functions.
"""


def get_sampling_function(sampling_method, noise_std, contain_method):
    """
    Define a sampling function for uniform sampling.
    """
    # Defining sampling strategies
    fn_sample_surface = sample_surface_simple
    fn_sample_near_surface = partial(
        sample_near_surface,
        noise_std=noise_std,
        contain_method=contain_method,
    )
    fn_sample_volume = partial(sample_volume, contain_method=contain_method)

    # Defining the sampling function
    sampling_fn = {
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
    }
    assert sampling_method in sampling_fn, "Invalid sampling method!"

    return sampling_fn[sampling_method]


def get_sampling_function_dist(sampling_method, face_dist, noise_std, contain_method):
    """
    Define a sampling function based on an input distribution.
    """
    sample_near_surface_weighted = partial(
        sample_distribution,
        face_dist=face_dist,
        noise_std=noise_std,
        contain_method=contain_method,
    )
    sample_near_surface_uniform = partial(
        sample_distribution,
        face_dist=face_dist,
        noise_std=noise_std,
        contain_method=contain_method,
    )

    sampling_fn = {
        "near_surface_weighted": sample_near_surface_weighted,
        "near_surface_weighted+uniform": partial(
            combine_samplings,
            sampling_fns=[
                sample_near_surface_weighted,
                sample_near_surface_uniform,
            ],
        ),
        "volume+near_surface_weighted": partial(
            combine_samplings,
            sampling_fns=[sample_volume, sample_near_surface_weighted],
        ),
        "volume+near_surface_weighted+uniform": partial(
            combine_samplings,
            sampling_fns=[
                sample_volume,
                sample_near_surface_weighted,
                sample_near_surface_uniform,
            ],
        ),
    }
    assert sampling_method in sampling_fn, "Invalid sampling method!"

    return sampling_fn[sampling_method]
