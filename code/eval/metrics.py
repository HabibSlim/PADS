"""
Compute evaluation metrics.
"""

import numpy as np
import torch
from losses.chamfer import chamfer_loss


"""
Defining all metrics.
"""


def l2_dist(x_gt, x_edited):
    """
    Compute the L2 distance between two latents.
    """
    return torch.norm(x_edited - x_gt, p=2)


def chamfer_reconstructed(pc_gt, pc_pred):
    """
    Compute the chamfer distance between the reconstructed edited and gt pointclouds.
    """
    return chamfer_loss(pc_gt, pc_pred, reduction="mean").mean()


def iou_occupancies(pred_occ, gt_occ):
    """
    Compute the IoU between predicted and ground-truth occupancy vectors/tensors.
    Supports both NumPy arrays and PyTorch tensors.
    """
    if isinstance(pred_occ, np.ndarray) and isinstance(gt_occ, np.ndarray):
        intersection = np.logical_and(pred_occ, gt_occ).sum()
        union = np.logical_or(pred_occ, gt_occ).sum()
        iou = (intersection + 1e-5) / (union + 1e-5)
    elif isinstance(pred_occ, torch.Tensor) and isinstance(gt_occ, torch.Tensor):
        intersection = torch.logical_and(pred_occ, gt_occ).sum().float()
        union = torch.logical_or(pred_occ, gt_occ).sum().float()
        iou = (intersection + 1e-5) / (union + 1e-5)
    else:
        raise TypeError(
            "Inputs must be either both NumPy arrays or both PyTorch tensors"
        )

    return iou
