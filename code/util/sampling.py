"""
Pointcloud sampling utilities.
"""

import torch
import trimesh


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


def sample_surface_tpp(mesh, num_points, device="cuda"):
    """
    Sample points on the surface of a mesh using Triangle Point Picking.
    CUDA-compatible sampling method.

    sample method:
    http://mathworld.wolfram.com/TrianglePointPicking.html
    """
    vertices, faces = mesh.torch_mesh()
    N = vertices.shape[0]

    # vertices = torch.tensor(vertices).float().to(device)
    # faces = torch.tensor(faces).long().to(device)

    weights, normal = face_areas_normals(vertices, faces)
    weights_sum = torch.sum(weights, dim=1)
    dist = torch.distributions.categorical.Categorical(
        probs=weights / weights_sum[:, None]
    )
    face_index = dist.sample((num_points,))

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
