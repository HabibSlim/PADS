"""
Core shape latent optimization functions.
"""

import torch
import torch.nn.functional as F
import util.s2vs as s2vs
import util.misc as misc

from datasets.sampling import (
    sample_distribution,
    sample_volume,
    combine_samplings,
)
from losses.rendering_loss import faces_scores, rendering_loss
from functools import partial


def error_distribution(gt_mesh, rec_mesh, squeeze_factor=0.5):
    """
    Compute the error distribution of a reconstructed mesh w.r.t. the ground-truth mesh,
    as a weighting over the faces of the mesh.
    """
    # Compute face-wise error scores
    f_scores = faces_scores(gt_mesh, rec_mesh, render_resolution=512, num_samples=32)

    # Sample faces
    weights = torch.pow(f_scores, squeeze_factor)
    weights_sum = torch.sum(weights)
    face_dist = torch.distributions.categorical.Categorical(
        probs=(weights / weights_sum).unsqueeze(0)
    )

    return face_dist


def optimize_latents(
    ae,
    shape_dataset,
    init_latents,
    object_id,
    *,
    accumulation_steps=1,
    max_iter=100,
    optimizer=None,
    lr=1e-3,
):
    """
    Optimize input latent codes w.r.t. a single object with optional gradient accumulation.
    """
    latents = init_latents.clone().detach().to(ae.device).requires_grad_(True)
    if optimizer is None:
        optimizer = torch.optim.Adam
    optimizer = optimizer([latents], lr=lr)

    # Defining shape iterators
    shape_it = iter(shape_dataset[object_id])

    # Main optimization loop
    iter_count = 0
    best_latents = latents.clone()
    best_r_loss = float("inf")
    while iter_count < max_iter:
        optimizer.zero_grad()
        accumulated_loss = 0

        for k in range(accumulation_steps):
            try:
                surface_points, occs = next(shape_it)
            except StopIteration:
                break

            surface_points = surface_points.to(ae.device)
            logits = s2vs.query_latents(
                ae, latents, surface_points, use_graph=True
            ).flatten()
            occs = occs.float().flatten().to(ae.device)

            loss = F.binary_cross_entropy_with_logits(logits, occs).mean()
            accumulated_loss += loss.item()

            # Accumulate gradients without stepping the optimizer
            loss.backward()

        # Step the optimizer
        optimizer.step()

        if iter_count % 10 == 0:
            # Evaluate rendering loss
            rec_mesh = s2vs.decode_latents(
                ae, misc.d_GPU(latents), grid_density=512, batch_size=128**3
            )
            mask_loss, depth_loss = rendering_loss(
                shape_dataset.mesh, rec_mesh, render_resolution=256
            )
            r_loss = depth_loss.item()

            if r_loss < best_r_loss:
                best_r_loss = r_loss
                best_latents = latents.clone()

            print(
                f"Iter {iter_count}: Average Loss {(accumulated_loss / accumulation_steps):.4f}, Rendering Loss {r_loss:.4f}"
            )

        iter_count += 1

    return best_latents.detach().cpu()


def refine_latents(
    ae,
    gt_mesh,
    init_latents,
    num_points,
    best_loss,
    *,
    accumulation_steps=1,
    max_iter=100,
    refresh_dist_every=25,
    optimizer=None,
    lr=1e-3,
    sampling_method="near_surface_weighted",
):
    """
    Optimize input latent codes by sampling high-error regions with a higher frequency.
    """

    def get_sampling_fn(face_dist):
        """
        Return the sampling function for the current error distribution.
        """
        sample_near_surface_weighted = partial(sample_distribution, face_dist=face_dist)
        sample_near_surface_uniform = partial(sample_distribution, face_dist=None)

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
        }[sampling_method]

        return sampling_fn

    latents = init_latents.clone().detach().to(ae.device).requires_grad_(True)
    if optimizer is None:
        optimizer = torch.optim.Adam
    optimizer = optimizer([latents], lr=lr)

    # Compute the error distribution
    face_dist = error_distribution(gt_mesh, s2vs.decode_latents(ae, latents))
    gt_mesh.set_grad(requires_grad=False)
    sampling_fn = get_sampling_fn(face_dist)

    # Main optimization loop
    iter_count = 0
    best_latents = latents.clone()
    best_r_loss = best_loss
    while iter_count < max_iter:
        optimizer.zero_grad()
        accumulated_loss = 0

        for k in range(accumulation_steps):
            surface_points, occs = sampling_fn(gt_mesh, num_points)
            surface_points = surface_points.to(ae.device)
            occs = occs.float().flatten().to(ae.device)

            logits = s2vs.query_latents(
                ae, latents, surface_points, use_graph=True
            ).flatten()

            loss = F.binary_cross_entropy_with_logits(logits, occs).mean()
            accumulated_loss += loss.item()

            # Accumulate gradients without stepping the optimizer
            loss.backward()

        # Step the optimizer
        optimizer.step()

        # Evaluate the rendering loss
        if iter_count % 10 == 0:
            rec_mesh = s2vs.decode_latents(
                ae, misc.d_GPU(latents), grid_density=512, batch_size=128**3
            )
            mask_loss, depth_loss = rendering_loss(
                gt_mesh, rec_mesh, render_resolution=256
            )
            r_loss = depth_loss.item()

            if r_loss < best_r_loss:
                best_r_loss = r_loss
                best_latents = latents.clone()

            print(
                f"Iter {iter_count}: Average Loss {(accumulated_loss / accumulation_steps):.4f}, Rendering Loss {r_loss:.4f}"
            )

        # Re-compute the error distribution
        if iter_count % refresh_dist_every == 0 and iter_count > 0:
            face_dist = error_distribution(gt_mesh, s2vs.decode_latents(ae, latents))
            gt_mesh.set_grad(requires_grad=False)
            sampling_fn = get_sampling_fn(face_dist)

        iter_count += 1

    return best_latents.detach().cpu()
