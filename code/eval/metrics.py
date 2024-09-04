"""
Compute evaluation metrics.
"""

import numpy as np
import kaolin
import torch


def l2_dist(x_gt, x_edited):
    """
    Compute the L2 distance between two latents.
    """
    return torch.norm(x_edited - x_gt, p=2)


@torch.inference_mode()
def chamfer_distance(x, y, backend="kaolin"):
    """
    Compute the chamfer distance between two point clouds.
    """
    assert backend in ["kaolin", "torch"], "Invalid backend"
    if backend == "kaolin":
        return kaolin.metrics.pointcloud.chamfer_distance(x, y)
    elif backend == "torch":
        bs, num_points, points_dim = x.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        diag_ind = torch.arange(0, num_points).to(x.device)
        rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
        ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
        P = rx.transpose(2, 1) + ry - 2 * zz
        return P.min(1)[0].mean(), P.min(2)[0].mean()


@torch.inference_mode()
def f_score(gt, pred, radius=0.01, eps=1e-08, backend="kaolin"):
    """
    Compute the F-Score between a predicted and ground-truth point cloud.
    """
    assert backend in ["kaolin"], "Invalid backend"
    if backend == "kaolin":
        return kaolin.metrics.pointcloud.f_score(gt, pred, radius, eps)


@torch.inference_mode()
def chamfer_distance_1D(x, y):
    """
    Compute the unidirectional chamfer distance from point cloud a to point cloud b.
    """
    xx = torch.sum(x**2, dim=2, keepdim=True)
    yy = torch.sum(y**2, dim=2, keepdim=True).transpose(1, 2)
    zz = torch.bmm(x, y.transpose(2, 1))
    P = xx + yy - 2 * zz
    return P.min(1)[0].mean()


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
