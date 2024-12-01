"""
Training and evaluation functions for the PartQueries-conditioned diffusion model.
"""

import math
import sys

import torch
import util.misc as misc
import wandb
from losses.partvae import ScaleInvariantLoss, PartDropLoss
from losses.edmloss_part_assets import EDMLossPartAssets

PRINT_FREQ = 50


def get_losses():
    """
    Instantiate the losses.
    """
    return (
        EDMLossPartAssets(),
        ScaleInvariantLoss(),
        PartDropLoss(),
    )


class PairType:
    NO_ROT_PAIR = "rand_no_rot,rand_no_rot"
    PART_DROP = "part_drop,orig"


def forward_pass(
    pqdm, data_tuple, rec_loss, scale_inv_loss, part_drop_loss, num_samples=512
):
    """
    Compute a single forward pass of the model.
    """
    device = pqdm.device

    # Unpack the data tuple
    (
        pair_types,
        (l_a, bb_a, bb_l_a, part_pts_a, shape_cls_a, _),
        (l_b, bb_b, bb_l_b, part_pts_b, shape_cls_b, _),
    ) = data_tuple

    # Compute the mask from batch labels
    mask_a = (bb_l_a == -1).to(device)  # B x 24
    mask_b = (bb_l_b == -1).to(device)  # B x 24

    l_a, l_b = l_a.to(device), l_b.to(device)  # B x 8 x 512
    bb_a, bb_b = bb_a.to(device), bb_b.to(device)  # B x 24 x 4 x 3
    bb_l_a, bb_l_b = bb_l_a.to(device), bb_l_b.to(device)  # B x 24

    # L2 loss
    rec_loss_a, part_queries_a = rec_loss(
        pqdm,
        l_a,
        bb_a,
        bb_l_a,
        part_pts_a,
        shape_cls_a,
        mask_a,
        num_samples=num_samples,
    )
    rec_loss_b, part_queries_b = rec_loss(
        pqdm,
        l_b,
        bb_b,
        bb_l_b,
        part_pts_b,
        shape_cls_b,
        mask_b,
        num_samples=num_samples,
    )
    rec_loss_value = (rec_loss_a + rec_loss_b) / 2.0

    # if pair_types == PairType.NO_ROT_PAIR:
    #     inv_loss = scale_inv_loss(part_queries_a, part_queries_b, mask_a)
    # elif pair_types == PairType.PART_DROP:
    #     inv_loss = part_drop_loss(
    #         part_queries_a, part_queries_b, bb_a, bb_b, mask_a, mask_b
    #     )
    inv_loss = torch.tensor(0.0).to(device)

    return {
        "rec_loss": rec_loss_value,
        "inv_loss": inv_loss,
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

    # Disable training for the LatentArrayTransformer
    if args.freeze_dm:
        model.module.freeze_dm()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Epoch: [{}]".format(epoch)

    accum_iter = args.accum_iter
    optimizer.zero_grad()

    # Instantiate the losses
    rec_loss, scale_inv_loss, part_drop_loss = get_losses()

    data_seen = False
    for data_step, data_tuple in enumerate(
        metric_logger.log_every(data_loader, PRINT_FREQ, header)
    ):
        # Computing loss
        loss = forward_pass(
            pqdm=model,
            data_tuple=data_tuple,
            rec_loss=rec_loss,
            scale_inv_loss=scale_inv_loss,
            part_drop_loss=part_drop_loss,
            num_samples=args.num_part_samples,
        )

        total_loss = (
            args.rec_weight * loss["rec_loss"] + args.inv_weight * loss["inv_loss"]
        )
        data_seen = True

        # Stop training if loss explodes
        loss_value = total_loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

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

        # Log the losses
        loss_update = {
            "train_loss": float(loss_value),
            "inv_loss": float(loss["inv_loss"]),
        }
        metric_logger.update(**loss_update)

        # Log the losses to wandb
        if (
            global_rank == 0
        ):  # and ((data_step + 1) % accum_iter == 0 or args.debug_run):
            epoch_1000x = int((data_step / len(data_loader) + epoch) * 1000)
            wandb.log(
                {
                    "epoch_1000x": epoch_1000x,
                    "train_batch_loss": loss_update["train_loss"],
                    "inv_batch_loss": loss_update["inv_loss"],
                }
            )

        if args.debug_run:
            break

    assert data_seen, "No data seen in training loop"

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()

    # Print metric logger keys
    print("Averaged stats:", metric_logger)
    if global_rank == 0:
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": float(metric_logger.train_loss.global_avg),
                "inv_loss": float(metric_logger.inv_loss.global_avg),
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
    Evaluate the model.
    """
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"

    # Switch to evaluation mode
    model.eval()

    # Instantiate the losses
    rec_loss, scale_inv_loss, part_drop_loss = get_losses()

    for data_step, data_tuple in enumerate(
        metric_logger.log_every(data_loader, PRINT_FREQ, header)
    ):
        # Computing loss
        loss = forward_pass(
            pqdm=model,
            data_tuple=data_tuple,
            rec_loss=rec_loss,
            scale_inv_loss=scale_inv_loss,
            part_drop_loss=part_drop_loss,
        )
        total_loss = (
            args.rec_weight * loss["rec_loss"] + args.inv_weight * loss["inv_loss"]
        )

        # Stop training if loss explodes
        loss_value = total_loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # Log the losses
        loss_update = {
            "eval_loss": float(loss_value),
            "inv_loss": float(loss["inv_loss"]),
        }
        metric_logger.update(**loss_update)

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if global_rank == 0:
        wandb.log(
            {
                "epoch": epoch,
                "eval_loss": float(metric_logger.eval_loss.global_avg),
            }
        )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
