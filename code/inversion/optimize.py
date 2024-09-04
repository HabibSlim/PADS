"""
Core shape latent optimization functions.
"""

import torch
import torch.nn.functional as F
import util.s2vs as s2vs
import util.misc as misc

from datasets.sampling import get_sampling_function_dist
from losses.rendering_loss import faces_scores, rendering_loss


def evaluate_rendering_loss(
    ae,
    latents,
    gt_mesh,
    *,
    render_resolution=256,
    grid_density=512,
    batch_size=128**3,
    n_views=16,
):
    """
    Evaluate the rendering loss of a reconstructed shape w.r.t. the ground-truth shape.
    """
    # Evaluate rendering loss
    rec_mesh = s2vs.decode_latents(
        ae, misc.d_GPU(latents), grid_density=grid_density, batch_size=batch_size
    )
    mask_loss, depth_loss = rendering_loss(
        gt_mesh, rec_mesh, render_resolution=render_resolution, n_views=n_views
    )

    return depth_loss.item()


def error_distribution(gt_mesh, rec_mesh, squeeze_factor):
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
    gt_mesh = shape_dataset.get_mesh(object_id)

    # Main optimization loop
    iter_count = 0
    best_latents = latents.clone()
    best_r_loss = evaluate_rendering_loss(ae, latents, gt_mesh)

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
            r_loss = evaluate_rendering_loss(ae, latents, gt_mesh)

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
    near_surface_noise=0.05,
    accumulation_steps=1,
    max_iter=100,
    refresh_dist_every=25,
    optimizer=None,
    lr=1e-3,
    sampling_method="near_surface_weighted",
    squeeze_factor=0.5,
):
    """
    Optimize input latent codes by sampling high-error regions with a higher frequency.
    """

    latents = init_latents.clone().detach().to(ae.device).requires_grad_(True)
    if optimizer is None:
        optimizer = torch.optim.Adam
    optimizer = optimizer([latents], lr=lr)

    # Compute the error distribution
    face_dist = error_distribution(
        gt_mesh, s2vs.decode_latents(ae, latents), squeeze_factor=squeeze_factor
    )
    gt_mesh.set_grad(requires_grad=False)
    sampling_fn = get_sampling_function_dist(
        sampling_method=sampling_method,
        face_dist=face_dist,
        noise_std=near_surface_noise,
        contain_method="occnets",
    )

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
            r_loss = evaluate_rendering_loss(ae, latents, gt_mesh)

            if r_loss < best_r_loss:
                best_r_loss = r_loss
                best_latents = latents.clone()

            print(
                f"Iter {iter_count}: Average Loss {(accumulated_loss / accumulation_steps):.4f}, Rendering Loss {r_loss:.4f}"
            )

        # Re-compute the error distribution
        if iter_count % refresh_dist_every == 0 and iter_count > 0:
            face_dist = error_distribution(
                gt_mesh, s2vs.decode_latents(ae, latents), squeeze_factor=squeeze_factor
            )
            gt_mesh.set_grad(requires_grad=False)
            sampling_fn = get_sampling_function_dist(
                sampling_method=sampling_method,
                face_dist=face_dist,
                noise_std=near_surface_noise,
                contain_method="occnets",
            )

        iter_count += 1

    return best_latents.detach().cpu()
