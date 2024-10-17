"""
Mesh-related functions/classes.
"""

import math
import fast_simplification
import kaolin as kal
import numpy as np
import torch
import trimesh
from datasets.sampling import face_areas


def show_side_by_side(*meshes, flip_front_to_back=False, viewer=None, get_mesh=False):
    """
    Show any number of meshes side by side.
    """
    if len(meshes) == 0:
        raise ValueError("At least one mesh must be provided.")

    # Clone the meshes
    cloned_meshes = [mesh.copy() for mesh in meshes]

    # Calculate the translation for each mesh
    num_meshes = len(cloned_meshes)
    translations = np.linspace(-1, 1, num_meshes)

    # Apply translations
    for mesh, translation in zip(cloned_meshes, translations):
        mesh.apply_translation([translation, 0, 0])

    if get_mesh:
        return trimesh.util.concatenate(cloned_meshes)

    # Combine them into a single scene
    scene = trimesh.Scene(cloned_meshes)
    return scene.show(viewer)


class CUDAMesh:
    """
    CUDA-compatible kaolin+trimesh wrapper class.
    """

    def __init__(self, vertices, faces):
        if isinstance(vertices, np.ndarray):
            vertices = torch.tensor(vertices).float()
        if isinstance(faces, np.ndarray):
            faces = torch.tensor(faces.astype(np.int64)).long()
        if torch.cuda.is_available():
            vertices = vertices.cuda()
            faces = faces.cuda()
        self.kaolin_mesh = kal.rep.SurfaceMesh(vertices, faces)
        self.trimesh_mesh = trimesh.Trimesh(
            vertices.squeeze().cpu().numpy(), faces.squeeze().cpu().numpy()
        )
        self.face_dist = None

    def batched_mesh(self, B, padding_value=0):
        """
        Batch the mesh vertices.
        """
        N, C = self.vertices.shape

        # Calculate K as the closest multiple of B to N
        K = math.ceil(N / B) * B // B

        # Calculate required padding
        total_elements = B * K
        padding_size = max(0, total_elements - N)

        # Pad the tensor if necessary
        if padding_size > 0:
            padding = torch.full(
                (padding_size, C),
                padding_value,
                dtype=self.vertices.dtype,
                device=self.vertices.device,
            )
            padded_tensor = torch.cat([self.vertices, padding], dim=0)
        else:
            padded_tensor = self.vertices

        # Reshape the tensor
        return padded_tensor.reshape(B, K, C), self.faces

    def copy(self):
        """
        Copy the trimesh instance and return it.
        """
        return self.trimesh_mesh.copy()

    def export(self, obj_file):
        """
        Export the mesh to an obj file.
        """
        return self.trimesh_mesh.export(obj_file)

    def load(obj_file, process=False):
        """
        Load a mesh from an obj file.
        """
        if process:
            # First load through trimesh
            trimesh_mesh = trimesh.load_mesh(obj_file, process=True)
            mesh = CUDAMesh.from_trimesh(trimesh_mesh)
            if torch.cuda.is_available():
                mesh.kaolin_mesh = mesh.kaolin_mesh.cuda()
            return mesh
        mesh = kal.io.obj.import_mesh(obj_file)
        if torch.cuda.is_available():
            mesh = mesh.cuda()
        return CUDAMesh(mesh.vertices, mesh.faces)

    def from_trimesh(trimesh_mesh):
        """
        Create a CUDAMesh instance from a trimesh object.
        """
        mesh = CUDAMesh(trimesh_mesh.vertices, trimesh_mesh.faces)
        if torch.cuda.is_available():
            mesh.kaolin_mesh = mesh.kaolin_mesh.cuda()
        return mesh

    def set_grad(self, requires_grad=True):
        """
        Toggle gradient computation for the vertices.
        """
        self.kaolin_mesh.vertices.requires_grad = requires_grad

    def get_grad(self):
        """
        Get the gradient of the vertices.
        """
        return self.kaolin_mesh.vertices.grad

    def show(self):
        """
        Create a trimesh object from the mesh and display it.
        """
        return self.trimesh_mesh.show()

    def apply_transform(self, transform):
        """
        Apply a transformation matrix to the mesh.
        """
        self.trimesh_mesh.apply_transform(transform)
        self.kaolin_mesh.vertices = torch.tensor(
            self.trimesh_mesh.vertices, dtype=self.kaolin_mesh.vertices.dtype
        ).to(self.kaolin_mesh.vertices.device)
        return self

    def recenter(self):
        """
        Recenter the mesh vertices around the origin.
        """
        # Recenter the trimesh mesh
        center = self.trimesh_mesh.bounding_box.centroid
        translation_mat = trimesh.transformations.translation_matrix(-center)

        # Apply the transformation
        self.trimesh_mesh.apply_transform(translation_mat)

        return self, translation_mat

    def to(self, device):
        """
        Move the mesh to the given device.
        """
        self.kaolin_mesh = self.kaolin_mesh.to(device)
        return self

    @property
    def face_distribution(self):
        """
        Compute a distribution over faces based on their surface areas.
        """
        if self.face_dist is None:
            areas = face_areas(self.vertices, self.faces)
            weights_sum = torch.sum(areas, dim=1)
            self.face_dist = torch.distributions.categorical.Categorical(
                probs=areas / weights_sum[:, None]
            )
        return self.face_dist

    @property
    def is_watertight(self):
        """
        Check if the mesh is watertight.
        """
        return self.trimesh_mesh.is_watertight

    @property
    def faces(self):
        return self.kaolin_mesh.faces

    @property
    def vertices(self):
        return self.kaolin_mesh.vertices.unsqueeze(0)


def decimate_mesh(mesh, factor):
    """
    Decimate the input mesh by the given factor using Fast Quadric Mesh Simplification.
    """
    vertices, faces = mesh.trimesh_mesh.vertices, mesh.trimesh_mesh.faces
    vertices_out, faces_out = fast_simplification.simplify(vertices, faces, factor)
    return CUDAMesh(vertices_out, faces_out)
