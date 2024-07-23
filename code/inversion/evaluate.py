"""
Evaluation functions for the shape inversion task.
"""

import torch
import util.s2vs as s2vs
from eval.metrics import iou_occupancies, chamfer_distance, f_score
from datasets.sampling import sample_surface_tpp
from util.contains.inside_mesh import is_inside
from losses.rendering_loss import rendering_loss


def evaluate_iou_only(ae, gt_mesh, latents, *, n_queries=2**21, query_method="occnets"):
    """
    Compute IoU between the original object and the reconstructed mesh.
    """
    # Sample uniformly n_queries points from 3D space
    point_queries = torch.rand((1, n_queries, 3)).cuda()

    # Get predicted occupancies for each point
    latents = latents.to(ae.device)
    pred_occs = s2vs.predict_occupancies(ae, latents, point_queries, n_queries)

    # Get ground-truth occupancy for each point
    gt_occs = is_inside(gt_mesh, point_queries, query_method=query_method)

    # Compute IoU between predicted and ground-truth occupancies
    iou = iou_occupancies(pred_occs, gt_occs)

    return iou.item()


def evaluate_reconstruction(
    ae,
    gt_mesh,
    rec_mesh,
    latents,
    *,
    n_queries=2**21,
    n_queries_chamfer=2**15,
    query_method="occnets",
):
    """
    Compute IoU and Chamfer distance between the original object and the reconstructed mesh.
    - IoU: Computed using [n_queries] random points sampled from the 3D sp  ace.
    - CD: Computed using [n_queries] points sampled from the object surface.
    """
    # Sample N_POINTS points from gt_mesh and rec_mesh
    pc_original = sample_surface_tpp(gt_mesh, n_queries_chamfer)
    pc_rec = sample_surface_tpp(rec_mesh, n_queries_chamfer)
    chamfer = chamfer_distance(pc_original, pc_rec)
    f_sc = f_score(gt=pc_original, pred=pc_rec)

    # Sample uniformly n_queries points from 3D space
    point_queries = torch.rand((1, n_queries, 3)).cuda()

    # Get predicted occupancies for each point
    latents = latents.to(ae.device)
    pred_occs = s2vs.predict_occupancies(ae, latents, point_queries, n_queries)

    # Get ground-truth occupancy for each point
    gt_occs = is_inside(gt_mesh, point_queries, query_method=query_method)

    # Compute IoU between predicted and ground-truth occupancies
    iou = iou_occupancies(pred_occs, gt_occs)

    # Compute rendering loss
    mask_loss, depth_loss = rendering_loss(gt_mesh, rec_mesh, render_resolution=256)

    results = {
        "iou": iou.item(),
        "chamfer": chamfer.item(),
        "f_score": f_sc.item(),
        "mask_loss": mask_loss.item(),
        "depth_loss": depth_loss.item(),
    }

    return results
