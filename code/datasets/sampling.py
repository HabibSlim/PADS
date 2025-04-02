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
def face_areas(vertices, faces):
    """
    Get face areas for a batch of meshes.
    """
    # print(vertices.shape, faces[:, 1].min(), faces[:, 1].max())
    vA = vertices[:, faces[:, 1], :]
    vA -= vertices[:, faces[:, 0], :]
    vB = vertices[:, faces[:, 2], :] - vertices[:, faces[:, 1], :]
    face_normals = torch.cross(
        vA,
        vB,
        dim=2,
    )
    face_areas = torch.norm(face_normals, dim=2)
    return 0.5 * face_areas


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


def get_bounding_sphere_radius(mesh):
    """
    Calculate the radius of a bounding sphere from a 3D bounding box vertices list.
    The sphere is centered at the origin (0,0,0).

    Args:
        vertices: List of (x,y,z) coordinates representing the bounding box vertices

    Returns:
        float: Radius of the bounding sphere
    """
    vertices = mesh.trimesh_mesh.bounding_box_oriented.vertices

    # Convert vertices to numpy array for easier computation
    vertices = np.array(vertices)

    # Calculate distances from origin to each vertex
    distances = np.sqrt(np.sum(vertices**2, axis=1))

    # Return the maximum distance as the radius
    return np.max(distances)


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
            points = torch.tensor(points).float()
        return points.unsqueeze(0)


def sample_cube(num_points):
    """
    Sample points in the unit cube.
    """
    points = torch.randn((1, num_points, 3))
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
    points += torch.randn(num_points, 3).to(points.device) * noise_std

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


def sample_sphere(num_points, bounding_radii=1.0):
    """
    Sample points uniformly by volume within a unit sphere.

    Args:
        num_points (int): Number of points to sample

    Returns:
        torch.Tensor: Points with shape (1, num_points, 3)
    """
    # Generate random points in normal distribution
    points = torch.randn(1, num_points, 3)

    # Normalize to unit sphere surface
    points = points / torch.norm(points, dim=2, keepdim=True)

    # Generate radiuses with cubic root distribution for uniform volume sampling
    radii = torch.rand(1, num_points, 1).pow(1 / 3)

    # Scale points by radii and bounding radii
    points = points * radii * bounding_radii

    return points


def sample_surface_simple(mesh, num_points, get_occ=True):
    """
    Randomly sample points on the mesh surface.
    """
    points = sample_surface(mesh, num_points)
    if get_occ:
        return points, torch.ones(num_points)
    return points


"""
Helper functions for unbalanced and balanced sampling.
"""


def _unbalanced_volume_sampling(
    mesh, num_points, radius, hash_resolution, contain_method, get_occ=True
):
    """Helper function for unbalanced volume sampling."""
    points = sample_sphere(num_points, radius).to(mesh.device)
    # Ensure points are [N, 3]
    points = points.reshape(-1, 3)
    if get_occ:
        occs = is_inside(mesh, points, hash_resolution, contain_method)
        points = points.unsqueeze(0)  # Add batch dimension [1, N, 3]
        return points, occs
    return points.unsqueeze(0)  # Add batch dimension [1, N, 3]


def _unbalanced_surface_sampling(
    mesh, num_points, noise_std, hash_resolution, contain_method
):
    """Helper function for unbalanced surface sampling."""
    points = sample_surface(mesh, num_points).to(mesh.device)
    # Ensure points are [N, 3]
    points = points.reshape(-1, 3)
    points += torch.randn(num_points, 3, device=mesh.device) * noise_std
    occs = is_inside(mesh, points, hash_resolution, contain_method)
    points = points.unsqueeze(0)  # Add batch dimension [1, N, 3]
    return points, occs


def _balanced_sampling(mesh, num_points, sampler_fn, sampler_args):
    """
    Generic balanced sampling procedure.
    """
    target = num_points // 2
    inside_points = torch.empty((target, 3), dtype=torch.float, device=mesh.device)
    outside_points = torch.empty((target, 3), dtype=torch.float, device=mesh.device)
    inside_count = 0
    outside_count = 0
    max_iterations = 5000

    for _ in range(max_iterations):
        if inside_count == target and outside_count == target:
            break

        batch_points, batch_occs = sampler_fn(mesh, 2 * target, *sampler_args)
        # Ensure points are [N, 3] and occs are [N]
        batch_points = batch_points.reshape(-1, 3)
        batch_occs = batch_occs.reshape(-1)

        # Fill inside buffer
        new_inside = batch_points[batch_occs]
        points_to_add = min(len(new_inside), target - inside_count)
        inside_points[inside_count : inside_count + points_to_add] = new_inside[
            :points_to_add
        ]
        inside_count += points_to_add

        # Fill outside buffer
        new_outside = batch_points[~batch_occs]
        points_to_add = min(len(new_outside), target - outside_count)
        outside_points[outside_count : outside_count + points_to_add] = new_outside[
            :points_to_add
        ]
        outside_count += points_to_add

    if inside_count != target or outside_count != target:
        # Pad inside points if needed
        if inside_count < target:
            repeat_count = target - inside_count
            pad_indices = torch.randint(
                0, inside_count, (repeat_count,), device=mesh.device
            )
            inside_points[inside_count:] = inside_points[pad_indices]

        # Pad outside points if needed
        if outside_count < target:
            repeat_count = target - outside_count
            pad_indices = torch.randint(
                0, outside_count, (repeat_count,), device=mesh.device
            )
            outside_points[outside_count:] = outside_points[pad_indices]

    points = torch.cat([inside_points, outside_points])
    occs = torch.cat(
        [
            torch.ones(target, dtype=torch.bool, device=mesh.device),
            torch.zeros(target, dtype=torch.bool, device=mesh.device),
        ]
    )

    perm = torch.randperm(num_points, device=mesh.device)
    points = points[perm].unsqueeze(0)  # Add batch dimension [1, N, 3]
    occs = occs[perm]  # Keep occupancies as [N]
    return points, occs


@torch.inference_mode()
def sample_volume_sphere(
    mesh,
    num_points,
    hash_resolution=512,
    contain_method="occnets",
    get_occ=True,
    balance=False,
):
    """
    Uniformly sample points by volume in the unit sphere.
    Return occupancies for the input mesh.
    """
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    if not get_occ and balance:
        raise ValueError("balance=True requires get_occ=True")

    radius = get_bounding_sphere_radius(mesh)

    if not balance:
        return _unbalanced_volume_sampling(
            mesh, num_points, radius, hash_resolution, contain_method, get_occ
        )

    sampler_args = (
        radius,
        hash_resolution,
        contain_method,
        True,
    )  # Always use get_occ=True for balanced
    result = _balanced_sampling(
        mesh, num_points, _unbalanced_volume_sampling, sampler_args
    )

    if result is None:  # Fallback to unbalanced if balanced sampling failed
        return _unbalanced_volume_sampling(
            mesh, num_points, radius, hash_resolution, contain_method, get_occ
        )
    return result


@torch.inference_mode()
def sample_near_surface(
    mesh,
    num_points,
    noise_std=0.05,
    hash_resolution=512,
    contain_method="occnets",
    get_occ=True,
    balance=False,
):
    """
    Randomly sample points close to the mesh surface.
    """
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")

    if not balance:
        return _unbalanced_surface_sampling(
            mesh, num_points, noise_std, hash_resolution, contain_method
        )

    sampler_args = (noise_std, hash_resolution, contain_method)
    result = _balanced_sampling(
        mesh, num_points, _unbalanced_surface_sampling, sampler_args
    )

    if result is None:  # Fallback to unbalanced if balanced sampling failed
        return _unbalanced_surface_sampling(
            mesh, num_points, noise_std, hash_resolution, contain_method
        )
    return result


def combine_samplings(mesh, num_points, sampling_fns):
    """
    Sample points using multiple sampling functions.
    """
    points, occs = [], []

    for k, sampling_fn in enumerate(sampling_fns):
        p, o = sampling_fn(mesh, num_points // len(sampling_fns), get_occ=True)
        o = o.to(p.device)
        points.append(p)
        occs.append(o)
    return torch.cat(points), torch.cat(occs)


def normalize_pc(points, method="max"):
    """
    Normalize a point cloud to [-1, 1] range with two methods:
    - method="max": normalize using maximum extent across all axes (partial cube filling)
    - method="per_axis": normalize each axis independently (full cube filling)
    Both methods center the point cloud at the origin.

    Args:
        points: Point cloud as torch.Tensor or numpy.ndarray with shape [..., 3]
        method: "max" or "per_axis"
    Returns:
        Normalized point cloud in same format as input
    """
    if isinstance(points, torch.Tensor):
        mins = torch.min(points, dim=-2)[0]
        maxs = torch.max(points, dim=-2)[0]
        center = (mins + maxs) / 2
        centered = points - center.unsqueeze(-2)

        extents = maxs - mins
        if method == "max":
            # Use maximum extent across all axes
            scale = torch.max(extents)
        elif method == "per_axis":
            # Scale each axis independently
            scale = extents.unsqueeze(-2)
        else:
            raise ValueError(f"Unknown method: {method}")

        return centered / scale

    else:  # numpy array
        mins = np.min(points, axis=-2)
        maxs = np.max(points, axis=-2)
        center = (mins + maxs) / 2
        centered = points - center[..., np.newaxis, :]

        extents = maxs - mins
        if method == "max":
            # Use maximum extent across all axes
            scale = np.max(extents)
        elif method == "per_axis":
            # Scale each axis independently
            scale = extents[..., np.newaxis, :]
        else:
            raise ValueError(f"Unknown method: {method}")

        return centered / scale


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
        "surface+near_surface": partial(
            combine_samplings,
            sampling_fns=[fn_sample_surface, fn_sample_near_surface],
        ),
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
