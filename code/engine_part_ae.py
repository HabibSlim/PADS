"""
Training and evaluation functions for the PartQueries-conditioned diffusion model.
"""

import math
import sys

import torch
import util.misc as misc
import util.lr_sched as lr_sched
import wandb

PRINT_FREQ = 50


def compute_metrics(outputs, labels, threshold):
    """
    Compute intersection over union and accuracy metrics.
    """
    pred = torch.zeros_like(outputs)
    pred[outputs >= threshold] = 1

    accuracy = (pred == labels).float().sum(dim=1) / labels.shape[1]
    accuracy = accuracy.mean()

    intersection = (pred * labels).sum(dim=1)
    union = (pred + labels).gt(0).sum(dim=1) + 1e-5

    iou = intersection * 1.0 / union
    iou = iou.mean()

    return accuracy, iou


def forward_pass(args, model, data_tuple, criterion):
    """
    Compute a single forward pass of the model.
    """
    device = model.device

    # Unpack data dict
    _, part_points, part_bbs, occ_points, occ_labels = data_tuple.values()

    # Move data to device
    part_points = part_points.to(device, non_blocking=True)
    part_bbs = part_bbs.to(device, non_blocking=True)
    occ_points = occ_points.to(device, non_blocking=True)
    occ_labels = occ_labels.to(device, non_blocking=True)

    # Forward pass
    outputs = model(part_points, part_bbs, occ_points)

    # Compute the KL loss
    if "kl" in outputs:
        loss_kl = outputs["kl"]
        loss_kl = torch.sum(loss_kl) / loss_kl.shape[0]
        loss_kl = args.kl_weight * loss_kl
    else:
        loss_kl = torch.tensor(0.0).to(device)

    # Chunk the outputs and labels
    outputs = outputs["logits"].squeeze()
    occ_labels = occ_labels.float()
    vol_outs, near_outs = outputs.chunk(2, dim=1)
    vol_labels, near_labels = occ_labels.chunk(2, dim=1)

    # Compute the near-surface loss
    loss_near = torch.tensor(0.0).to(device)
    if args.near_weights is not None:
        """
        In this case occupancies are split like follows
        With N the total number of query points
        [0:N//8], [N//8:N//4], [N//4:3N//8], [3N//8:N//2] : first 4 near surface point clouds
        these take weights from near_weights array in the same order
        [N//2, N] : the last 2 far volume point clouds. these take vol_weight
        """
        # Compute the near-surface loss
        for (k, near_weight), occ_chunk, occ_labels in zip(
            enumerate(args.near_weights),
            near_outs.chunk(4, dim=1),
            near_labels.chunk(4, dim=1),
        ):
            loss_near += near_weight * criterion(occ_chunk, occ_labels)
    else:
        loss_near = args.default_near_weights * criterion(near_outs, near_labels)

    # Compute the volume loss
    loss_vol = args.vol_weight * criterion(vol_outs, vol_labels)

    # Compute metrics for volume samples only
    acc, iou = compute_metrics(vol_outs, vol_labels, threshold=0.0)

    return {
        "loss_near": loss_near,
        "loss_vol": loss_vol,
        "loss_kl": loss_kl,
        "acc": acc,
        "iou": iou,
    }


def train_one_epoch(
    args,
    model,
    data_loader,
    optimizer,
    epoch,
    global_rank,
    loss_scaler,
    max_norm=5.0,
):
    """
    Train the model for one epoch.
    """
    model.train(True)
    torch.autograd.set_detect_anomaly(True)

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Epoch: [{}]".format(epoch)

    accum_iter = args.accum_iter
    optimizer.zero_grad()

    criterion = torch.nn.BCEWithLogitsLoss()

    data_seen = False
    for data_step, data_tuple in enumerate(
        metric_logger.log_every(data_loader, PRINT_FREQ, header)
    ):
        # Per-iteration (instead of per epoch) lr scheduler
        if data_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_step / len(data_loader) + epoch, args
            )

        # Computing loss
        output = forward_pass(
            args=args, model=model, data_tuple=data_tuple, criterion=criterion
        )
        total_loss = output["loss_vol"] + output["loss_near"] + output["loss_kl"]
        loss_value = total_loss.item()

        # Panic exit if loss is not finite
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        data_seen = True

        # Backward pass
        total_loss /= accum_iter
        loss_scaler(
            total_loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=False,
            update_grad=(data_step + 1) % accum_iter == 0,
        )

        if (data_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        torch.cuda.synchronize()

        # Log the losses and metrics
        loss_update = {
            "loss_vol": float(output["loss_vol"]),
            "loss_near": float(output["loss_near"]),
            "loss_kl": float(output["loss_kl"]),
            "acc": float(output["acc"]),
            "iou": float(output["iou"]),
        }
        metric_logger.update(**loss_update)

        # Log the losses to wandb
        if global_rank == 0:
            wandb.log(
                {
                    "data_step": epoch * len(data_loader) + data_step,
                    "loss_vol_batch": float(output["loss_vol"]),
                    "loss_near_batch": float(output["loss_near"]),
                    "loss_kl_batch": float(output["loss_kl"]),
                    "acc_batch": float(output["acc"]),
                    "iou_batch": float(output["iou"]),
                }
            )

    assert data_seen, "No data seen in training loop"

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()

    # Print metric logger keys
    print("Averaged stats:", metric_logger)
    if global_rank == 0:
        wandb.log(
            {
                "epoch": epoch,
                "loss_vol": float(metric_logger.loss_vol.global_avg),
                "loss_near": float(metric_logger.loss_near.global_avg),
                "loss_kl": float(metric_logger.loss_kl.global_avg),
                "acc": float(metric_logger.acc.global_avg),
                "iou": float(metric_logger.iou.global_avg),
            }
        )
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.inference_mode()
def evaluate(
    args,
    model,
    data_loader,
    epoch,
    global_rank,
):
    """
    Evaluate the model on the validation set.
    """
    model.eval()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Validation:"

    criterion = torch.nn.BCEWithLogitsLoss()

    data_seen = False
    for data_step, data_tuple in enumerate(
        metric_logger.log_every(data_loader, PRINT_FREQ, header)
    ):
        # Computing forward pass and metrics
        output = forward_pass(
            args=args, model=model, data_tuple=data_tuple, criterion=criterion
        )

        # Update running metrics
        loss_update = {
            "loss_vol": float(output["loss_vol"]),
            "loss_near": float(output["loss_near"]),
            "loss_kl": float(output["loss_kl"]),
            "acc": float(output["acc"]),
            "iou": float(output["iou"]),
        }
        metric_logger.update(**loss_update)

        data_seen = True

    assert data_seen, "No data seen in evaluation loop"

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()

    # Print metric logger keys
    print("Validation stats:", metric_logger)

    # Log validation metrics to wandb
    if global_rank == 0:
        wandb.log(
            {
                "epoch": epoch,
                "val_loss_vol": float(metric_logger.loss_vol.global_avg),
                "val_loss_near": float(metric_logger.loss_near.global_avg),
                "val_loss_kl": float(metric_logger.loss_kl.global_avg),
                "val_acc": float(metric_logger.acc.global_avg),
                "val_iou": float(metric_logger.iou.global_avg),
            }
        )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
