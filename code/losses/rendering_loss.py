"""
Defining the rendering loss used for shape inversion.
"""

import torch
import losses.diffrender as diffrender
from util.misc import CUDAMesh


def rendering_loss(gt_mesh, rec_mesh, render_resolution, n_views=16, device="cuda"):
    """
    Compute rendering loss between a GT mesh and a reconstructed mesh.

    gt_mesh: Target mesh
    rec_mesh: Reconstructed mesh
    render_resolution: Resolution of the rendered images for differentiable rendering
    """
    render_resolution = [render_resolution, render_resolution]

    # Sample random camera poses
    cameras = diffrender.get_random_camera_batch(
        n_views, iter_res=render_resolution, device=device
    )

    # Render GT mesh at sampled views
    target = diffrender.render_mesh(gt_mesh, cameras, render_resolution)

    # Extract and render FlexiCubes mesh
    rec = diffrender.render_mesh(rec_mesh, cameras, render_resolution)

    # Reconstruction loss
    mask_loss = (rec["mask"] - target["mask"]).abs().mean()  # mask loss
    depth_loss = (
        ((((rec["depth"] - target["depth"]) * target["mask"]) ** 2).sum(-1) + 1e-8)
        .sqrt()
        .mean()
    )

    return mask_loss, depth_loss


def min_max_norm(x):
    """
    Min-max normalization of a tensor.
    """
    return (x - x.min(0)[0]) / (
        x.max(0)[0] - x.min(0)[0]
    )  # (x - x.min()) / (x.max() - x.min())


def sigmoid(x, scale, shift):
    """
    Sigmoid function with shift and scaling (k)
    """
    return 1 / (1 + torch.exp(-scale * (x - shift)))


def vertices_gradients(
    gt_mesh, rec_mesh, render_resolution=256, get_face_weights=False
):
    """
    Compute the gradient of the rendering loss w.r.t. the vertices of the ground-truth mesh.
    """
    # Enable gradients for vertices
    gt_mesh.set_grad()
    rec_mesh.set_grad()

    # Compute rendering loss
    mask_loss, depth_loss = rendering_loss(gt_mesh, rec_mesh, render_resolution)

    # Get the gradient of the loss w.r.t. the vertices of the reconstructed mesh
    loss = mask_loss + depth_loss
    loss.backward()

    # Compute the weights for each face
    return gt_mesh.get_grad().clone().detach().abs()


def estimate_gradient(
    gt_mesh, rec_mesh, num_samples=32, render_resolution=256, get_face_weights=False
):
    """
    Monte-Carlo estiamtion of the gradient of the rendering loss
    w.r.t. the vertices of the ground-truth mesh.
    """
    # Averaging the gradients over multiple samples
    for i in range(num_samples):
        # Get the face weights
        vert_grads = vertices_gradients(
            gt_mesh,
            rec_mesh,
            render_resolution=render_resolution,
            get_face_weights=False,
        )

        # Accumulate the error
        if i != 0:
            vert_grads += vert_grads.detach()

    return vert_grads / num_samples


@torch.inference_mode()
def high_error_vertices(vert_grads, vertices, threshold=0.5, scale=12, shift=0.1):
    """
    Select high error vertices based on the gradient of the rendering loss.
    """
    vert_values = min_max_norm(vert_grads).mean(1)
    vert_values = sigmoid(vert_values, scale=scale, shift=shift)
    vert_values = min_max_norm(vert_values)

    # Get all vertices with a weight greater than the threshold
    return vertices[vert_values > threshold]


@torch.inference_mode()
def vert_score_to_faces(vert_scores, faces):
    """
    Transfer the scores of vertices to the faces they belong to.
    """
    # Compute face-wise weights
    face_scores = torch.index_select(vert_scores, 0, faces.flatten()).view(faces.shape)
    face_scores = min_max_norm(face_scores)
    return min_max_norm(face_scores.sum(dim=1))


def faces_scores(gt_mesh, rec_mesh, render_resolution=256, num_samples=32):
    """
    Compute the scores of the faces based on the multi-view rendering loss.
    """
    # Averaging the gradients over multiple samples
    vert_grads = estimate_gradient(
        gt_mesh, rec_mesh, num_samples=num_samples, render_resolution=256
    )

    with torch.inference_mode():
        vertices = gt_mesh.vertices.squeeze()
        selected_vertices = high_error_vertices(
            vert_grads, vertices, threshold=0.5, scale=12, shift=0.1
        )

        # Compute pairwise distances between selected vertices and all vertices
        dists = torch.cdist(vertices, selected_vertices)

        # Make a matrix of dim N x K giving the top-k distances for each vertex
        vert_scores = min_max_norm(torch.exp(-10 * dists).sum(dim=1))
        vert_scores = sigmoid(vert_scores, scale=4, shift=0.4)

    return vert_score_to_faces(vert_scores, gt_mesh.faces)


def scale_for_renders(mesh):
    """
    Rescale/translate meshes to fit in the rendering window.
    """
    vertices, faces = mesh.vertices, mesh.faces
    vertices = vertices.squeeze().clone()
    faces = faces.clone()
    vmin, vmax = vertices.min(dim=0)[0], vertices.max(dim=0)[0]
    scale = 1.8 / torch.max(vmax - vmin).item()
    vertices = vertices - (vmax + vmin) / 2  # Center mesh on origin
    vertices = vertices * scale  # Rescale to [-0.9, 0.9]
    return CUDAMesh(vertices, faces)
