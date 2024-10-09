"""
3DS2VS utility functions.
"""

import mcubes
import numpy as np
import torch
import torch.nn.functional as F
from contextlib import nullcontext
from util.mesh import CUDAMesh
import models.s2vs as s2vs


def load_model(model_name, ckpt_path, device, torch_compile=True):
    """
    Load a model from a file.
    """
    print("Loading autoencoder [%s]." % ckpt_path)

    # Instantiate autoencoder
    ae = s2vs.__dict__[model_name]()
    ae.load_state_dict(torch.load(ckpt_path, map_location="cpu")["model"])

    # Compile using torch.compile
    if torch_compile:
        ae = torch.compile(ae, mode="max-autotune")

    return ae.to(device)


def batch_slices(total, batch_size):
    """
    Batch slices generator.
    """
    start = 0
    while start < total:
        end = min(start + batch_size, total)
        yield start, end
        start = end


def get_grid(grid_density=128):
    """
    Get a 3D grid of points.
    """
    x = np.linspace(-1, 1, grid_density + 1)
    y = np.linspace(-1, 1, grid_density + 1)
    z = np.linspace(-1, 1, grid_density + 1)
    xv, yv, zv = np.meshgrid(x, y, z)
    grid = (
        torch.from_numpy(np.stack([xv, yv, zv]).astype(np.float32))
        .view(3, -1)
        .transpose(0, 1)[None]
    )
    return grid


def query_latents(ae, latent, point_queries, batch_size=None, use_graph=False):
    """
    Query latents using a set of point queries.
    Optionally, record a computation graph for backpropagation.
    """
    with nullcontext() if use_graph else torch.inference_mode():
        num_samples = point_queries.shape[1]

        if batch_size is None or num_samples <= batch_size:
            logits = ae.decode(latent, point_queries)
            return logits

        logits = torch.cat(
            [
                ae.decode(latent, point_queries[:, start_idx:end_idx, :])
                for start_idx, end_idx in batch_slices(num_samples, batch_size)
            ],
            dim=1,
        )

        return logits


@torch.inference_mode()
def query_latents_grid(ae, latent, grid_density, batch_size=None):
    """
    Query latents on a regular 3D grid.
    """
    sample_grid = get_grid(grid_density).to(latent.device)
    return query_latents(ae, latent, sample_grid, batch_size)


@torch.inference_mode()
def predict_occupancies(ae, latents, point_queries, n_queries):
    """
    Predict occupancies for each query point using the autoencoder.
    """
    logits = query_latents(ae, latents, point_queries, batch_size=128**3)

    # Apply sigmoid to logits to get occupancy probabilities
    logits = F.sigmoid(logits)
    pred_occs = (logits > 0.5).reshape(n_queries)

    return pred_occs


def decode_latents(ae, latent, grid_density=128, batch_size=None, smooth_volume=False):
    """
    Decode latents to a mesh using marching cubes.
    """
    B, N, D = latent.shape
    assert B == 1, "Batch size must be 1."

    with torch.inference_mode():
        logits = query_latents_grid(ae, latent, grid_density, batch_size)
        volume = (
            logits.view(grid_density + 1, grid_density + 1, grid_density + 1)
            .permute(1, 0, 2)
            .cpu()
            .numpy()
        )
        if smooth_volume:
            volume = mcubes.smooth(volume)
        verts, faces = mcubes.marching_cubes(volume, 0)
        gap = 2.0 / grid_density
        verts *= gap
        verts -= 1
    return CUDAMesh(verts, faces)


@torch.inference_mode()
def encode_pc(ae, pc, fps_sampling=False):
    """
    Encode a point cloud using the autoencoder.
    """
    new_pc = pc
    if pc.shape[1] > ae.num_inputs:
        if fps_sampling:
            ratio = ae.num_inputs / pc.shape[1]
            new_pc = s2vs.fps_subsample(pc, ratio)
        else:
            new_pc = pc[:, np.random.choice(pc.shape[1], ae.num_inputs, replace=False)]
    _, x_a = ae.encode(new_pc)
    return x_a


@torch.inference_mode()
def encode_decode(ae, pc, grid_density=256, batch_size=None):
    """
    Encode and decode a point cloud.
    """
    x_a = encode_pc(ae, pc)
    mesh = decode_latents(ae, x_a, grid_density, batch_size)
    return mesh, x_a
