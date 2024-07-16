"""
Test if points are inside a mesh using various methods.
"""

import kaolin
import torch
import numpy as np
import point_cloud_utils as pcu
from .triangle_hash import TriangleHash as _TriangleHash


def is_inside(mesh, points, hash_resolution=512, query_method="occnets"):
    """
    Check if points are inside the mesh.
    """
    assert query_method in [
        "occnets",
        "trimesh",
        "pcu",
        "kaolin",
    ], "Invalid query method"

    if query_method != "kaolin":
        points = points.squeeze()

    if query_method == "occnets":
        intersector = MeshIntersector(mesh, hash_resolution)
        contains = intersector.query(points)
    elif query_method == "trimesh":
        # Use trimesh to check if points are inside the mesh
        contains = mesh.contains(points)
    elif query_method == "pcu":
        v, f = mesh.vertices, mesh.faces
        sdf, fid, bc = pcu.signed_distance_to_mesh(points, v, f)
        contains = sdf < 0
    elif query_method == "kaolin":
        return kaolin.ops.mesh.check_sign(
            mesh.vertices, mesh.faces, points, hash_resolution=hash_resolution
        )

    return contains


class MeshIntersector:
    """
    MeshIntersector class to check if points are inside a mesh.
    """

    def __init__(self, mesh, resolution=512, device="cuda"):
        triangles = mesh.vertices.squeeze()[mesh.faces].double()
        n_tri = triangles.shape[0]

        self.resolution = resolution
        self.bbox_min = triangles.reshape(3 * n_tri, 3).min(dim=0)[0]
        self.bbox_max = triangles.reshape(3 * n_tri, 3).max(dim=0)[0]

        # Translate and scale it to [0.5, self.resolution - 0.5]^3
        self.scale = (resolution - 1) / (self.bbox_max - self.bbox_min)
        self.translate = 0.5 - self.scale * self.bbox_min

        self._triangles = triangles = self.rescale(triangles).to(device)

        triangles2d = triangles[:, :, :2].to(device)
        self._tri_intersector2d = TriangleIntersector2d(triangles2d, resolution)

        self.scale = self.scale.to(device)
        self.translate = self.translate.to(device)

    def query(self, points):
        # Rescale points
        points = self.rescale(points)

        # Placeholder result with no hits we'll fill in later
        contains = torch.zeros(len(points), dtype=torch.bool).to(points.device)

        # Cull points outside of the axis aligned bounding box
        # this avoids running ray tests unless points are close
        inside_aabb = torch.all((0 <= points) & (points <= self.resolution), dim=1)
        if not inside_aabb.any():
            return contains

        # Only consider points inside bounding box
        mask = inside_aabb
        points = points[mask]

        # Compute intersection depth and check order
        points_indices, tri_indices = self._tri_intersector2d.query(points[:, :2])

        triangles_intersect = self._triangles[tri_indices]
        points_intersect = points[points_indices]

        depth_intersect, abs_n_2 = self.compute_intersection_depth(
            points_intersect, triangles_intersect
        )

        # Count number of intersections in both directions
        smaller_depth = depth_intersect >= points_intersect[:, 2] * abs_n_2
        bigger_depth = depth_intersect < points_intersect[:, 2] * abs_n_2
        points_indices_0 = points_indices[smaller_depth]
        points_indices_1 = points_indices[bigger_depth]

        nintersect0 = torch.bincount(points_indices_0, minlength=points.shape[0])
        nintersect1 = torch.bincount(points_indices_1, minlength=points.shape[0])

        # Check if point contained in mesh
        contains1 = torch.fmod(nintersect0, 2) == 1
        contains2 = torch.fmod(nintersect1, 2) == 1
        # if (contains1 != contains2).any():
        #     print("Warning: contains1 != contains2 for some points.")
        contains[mask] = contains1 & contains2
        return contains

    def compute_intersection_depth(self, points, triangles):
        t1 = triangles[:, 0, :]
        t2 = triangles[:, 1, :]
        t3 = triangles[:, 2, :]

        v1 = t3 - t1
        v2 = t2 - t1

        normals = torch.cross(v1, v2)
        alpha = torch.sum(normals[:, :2] * (t1[:, :2] - points[:, :2]), dim=1)

        n_2 = normals[:, 2]
        t1_2 = t1[:, 2]
        s_n_2 = torch.sign(n_2)
        abs_n_2 = torch.abs(n_2)

        mask = abs_n_2 != 0

        depth_intersect = (
            torch.full((points.shape[0],), float("nan")).double().to(points.device)
        )
        depth_intersect[mask] = t1_2[mask] * abs_n_2[mask] + alpha[mask] * s_n_2[mask]

        return depth_intersect, abs_n_2

    def rescale(self, array):
        array = self.scale * array + self.translate
        return array


class TriangleIntersector2d:
    """
    TriangleIntersector2d class to check if 2D points are inside a triangle.
    """

    def __init__(self, triangles, resolution=128):
        self.triangles = triangles
        self.tri_hash = _TriangleHash(np.array(triangles.cpu()), resolution)

    def query(self, points):
        point_indices, tri_indices = self.tri_hash.query(np.array(points.cpu()))
        point_indices = torch.tensor(point_indices, dtype=torch.int64).to(points.device)
        tri_indices = torch.tensor(tri_indices, dtype=torch.int64).to(points.device)

        points = points[point_indices]
        triangles = self.triangles[tri_indices]
        mask = self.check_triangles(points, triangles)
        point_indices = point_indices[mask]
        tri_indices = tri_indices[mask]
        return point_indices, tri_indices

    def check_triangles(self, points, triangles):
        contains = torch.zeros(points.shape[0], dtype=torch.bool).to(points.device)
        A = triangles[:, :2] - triangles[:, 2:]
        A = A.transpose(1, 2)
        y = points - triangles[:, 2]

        detA = A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0]

        mask = torch.abs(detA) != 0.0
        A = A[mask]
        y = y[mask]
        detA = detA[mask]

        s_detA = torch.sign(detA)
        abs_detA = torch.abs(detA)

        u = (A[:, 1, 1] * y[:, 0] - A[:, 0, 1] * y[:, 1]) * s_detA
        v = (-A[:, 1, 0] * y[:, 0] + A[:, 0, 0] * y[:, 1]) * s_detA

        sum_uv = u + v
        contains[mask] = (
            (0 < u)
            & (u < abs_detA)
            & (0 < v)
            & (v < abs_detA)
            & (0 < sum_uv)
            & (sum_uv < abs_detA)
        )
        return contains
