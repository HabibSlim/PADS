"""
Training and evaluation functions for the PartVAE model.
"""

import math
import sys

import torch
from torch.nn import functional as F
import util.misc as misc
import wandb
from losses.partvae import KLRecLoss, ScaleInvariantLoss, PartDropLoss

PRINT_FREQ = 50


class PairType:
    NO_ROT_PAIR = "rand_no_rot,rand_no_rot"
    PART_DROP = "part_drop,orig"


def forward_pass(
    pvae,
    data_tuple,
    kl_rec_loss,
    scale_inv_loss,
    part_drop_loss,
    pair_types,
):
    """
    Compute a single forward pass of the model.
    """
    # Unpack the data tuple
    pair_types, (l_a, bb_a, bb_l_a, meta_a), (l_b, bb_b, bb_l_b, meta_b) = data_tuple
    device = pvae.device

    # Compute the mask from batch labels
    mask_a = (bb_l_a != -1).to(device)  # B x 24
    mask_b = (bb_l_b != -1).to(device)  # B x 24

    l_a, l_b = l_a.to(device), l_b.to(device)  # B x 24 x 512
    bb_a, bb_b = bb_a.to(device), bb_b.to(device)  # B x 24 x 4 x 3
    bb_l_a, bb_l_b = bb_l_a.to(device), bb_l_b.to(device)  # B x 24

    # Forward passes
    logits_a, kl_a, part_latents_a = pvae(
        latents=l_a, part_bbs=bb_a, part_labels=bb_l_a, batch_mask=mask_a
    )
    logits_b, kl_b, part_latents_b = pvae(
        latents=l_b, part_bbs=bb_b, part_labels=bb_l_b, batch_mask=mask_b
    )

    # KL Reg loss
    kl_reg = kl_rec_loss(kl_a) + kl_rec_loss(kl_b)

    # L2 loss
    rec_loss = F.mse_loss(logits_a, l_a) + F.mse_loss(logits_b, l_b)

    if pair_types == PairType.NO_ROT_PAIR:
        inv_loss = scale_inv_loss(part_latents_a, part_latents_b, mask_a)
    elif pair_types == PairType.PART_DROP:
        inv_loss = part_drop_loss(
            part_latents_a, part_latents_b, bb_a, bb_b, mask_a, mask_b
        )

    return {
        "kl_reg": kl_reg,
        "rec_loss": rec_loss,
        "inv_loss": inv_loss,
    }


def get_losses():
    """
    Instantiate the losses.
    """
    return (
        KLRecLoss(),
        ScaleInvariantLoss(),
        PartDropLoss(),
    )


def train_one_epoch(
    args,
    model,
    data_loader,
    optimizer,
    epoch,
    global_rank,
):
    """
    Train the model for one epoch.
    """
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Epoch: [{}]".format(epoch)

    accum_iter = args.accum_iter
    optimizer.zero_grad()

    # Instantiate the losses
    kl_loss, scale_inv_loss, part_drop_loss = get_losses()

    data_seen = False
    for data_step, data_tuple in enumerate(
        metric_logger.log_every(data_loader, PRINT_FREQ, header)
    ):
        # Computing loss
        loss = forward_pass(
            pvae=model,
            data_tuple=data_tuple,
            kl_rec_loss=kl_loss,
            scale_inv_loss=scale_inv_loss,
            part_drop_loss=part_drop_loss,
            pair_types=PairType.PART_DROP,
        )
        total_loss = (
            args.kl_weight * loss["kl_reg"]
            + args.rec_weight * loss["rec_loss"]
            + args.inv_weight * loss["inv_loss"]
        )
        data_seen = True

        # Stop training if loss explodes
        loss_value = total_loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # Backward pass
        total_loss /= accum_iter
        total_loss.backward()
        if (data_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        torch.cuda.synchronize()

        # Log the losses
        loss_update = {
            "train_loss": float(loss_value),
            "kl_reg": float(loss["kl_reg"].item()),
            "rec_loss": float(loss["rec_loss"].item()),
            "inv_loss": float(loss["inv_loss"].item()),
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
                    "train_kl_reg_batch_loss": loss_update["kl_reg"],
                    "train_rec_batch_loss": loss_update["rec_loss"],
                    "train_inv_batch_loss": loss_update["inv_loss"],
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
                "train_kl_reg_loss": float(metric_logger.kl_reg.global_avg),
                "train_rec_loss": float(metric_logger.rec_loss.global_avg),
                "train_inv_loss": float(metric_logger.inv_loss.global_avg),
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
    kl_loss, scale_inv_loss, part_drop_loss = get_losses()

    for data_step, data_tuple in enumerate(
        metric_logger.log_every(data_loader, PRINT_FREQ, header)
    ):
        # Computing loss
        loss = forward_pass(
            pvae=model,
            data_tuple=data_tuple,
            kl_rec_loss=kl_loss,
            scale_inv_loss=scale_inv_loss,
            part_drop_loss=part_drop_loss,
            pair_types=PairType.PART_DROP,
        )
        total_loss = (
            args.kl_weight * loss["kl_reg"]
            + args.rec_weight * loss["rec_loss"]
            + args.inv_weight * loss["inv_loss"]
        )

        # Stop training if loss explodes
        loss_value = total_loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # Log the losses
        loss_update = {
            "eval_loss": float(loss_value),
            "eval_kl_reg": float(loss["kl_reg"].item()),
            "eval_rec": float(loss["rec_loss"].item()),
            "eval_inv": float(loss["inv_loss"].item()),
        }
        metric_logger.update(**loss_update)

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if global_rank == 0:
        wandb.log(
            {
                "epoch": epoch,
                "eval_loss": float(metric_logger.eval_loss.global_avg),
                "eval_kl_reg": float(metric_logger.eval_kl_reg.global_avg),
                "eval_rec": float(metric_logger.eval_rec.global_avg),
                "eval_inv": float(metric_logger.eval_inv.global_avg),
            }
        )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
