"""
Utility functions to apply edits to batches of shapes.
"""
import torch


def apply_edit(net, x_a, _, embed_ab):
    """
    Apply the edit to the latent vector.
    """
    # Reshape from (B, D, K) to (B, M)
    x_a = x_a.flatten(1)
    embed_ab = embed_ab.flatten(1)

    # Concatenate the latent vector with the embedding
    edit_vec = net(x_a, embed_ab)

    # Add the edit vector to the latent vector
    return edit_vec + x_a


def apply_edit__oracle_mag(net, x_a, x_b, embed_ab):
    # Reshape from (B, D, K) to (B, M)
    x_a = x_a.flatten(1)
    embed_ab = embed_ab.flatten(1)

    # Concatenate the latent vector with the embedding
    edit_dir, magnitude = net.forward_decoupled(x_a, embed_ab)
    gt_vec = x_b - x_a
    opt_magnitude = torch.zeros_like(magnitude)
    for i in range(x_a.shape[0]):
        opt_magnitude[i] = solve_eq(a=edit_dir[i], b=gt_vec[i])
    edit_vec = edit_dir * opt_magnitude

    # Add the edit vector to the latent vector
    return edit_vec + x_a


def apply_edit__oracle_dir(net, x_a, x_b, embed_ab):
    # Reshape from (B, D, K) to (B, M)
    x_a = x_a.flatten(1)
    embed_ab = embed_ab.flatten(1)

    # Concatenate the latent vector with the embedding
    edit_dir, magnitude = net.forward_decoupled(x_a, embed_ab)
    gt_vec = x_b - x_a
    opt_direction = torch.zeros_like(edit_dir)
    for i in range(x_a.shape[0]):
        opt_direction[i] = gt_vec[i] / (torch.norm(gt_vec[i]) + 1e-8)

    # Add the edit vector to the latent vector
    return opt_direction * magnitude + x_a


def solve_eq(a, b):
    """
    Minimize ||x*a - b||^2, where x is a scalar, and a and b are fixed vectors.
    """
    return (b * a).sum() / (a * a).sum()


def apply_iterated_edits(
    model, ae_model, embeds_a, embeds_b, embeds_text, oracle_type="none", n_points=4096
):
    """
    Apply the edits iteratively.
    """
    if oracle_type == "none":
        edit_fn = apply_edit
    elif oracle_type == "mag":
        edit_fn = apply_edit__oracle_mag
    elif oracle_type == "dir":
        edit_fn = apply_edit__oracle_dir

    # Move all the garbage to CUDA
    x_a = embeds_a.cuda()
    x_b = embeds_b.cuda()
    embeds_text = embeds_text.cuda()

    x_b_edited = edit_fn(model, x_a, x_b, embeds_text)

    # Decode the batch
    b_size = x_b.shape[0]

    with torch.inference_mode():
        rec = ae_model.decoder(x_b_edited).reshape([b_size, n_points, 3]).cuda()
        rec_gt = ae_model.decoder(x_b).reshape([b_size, n_points, 3]).cuda()

    return (rec, rec_gt), (x_b_edited, x_b)
