"""
Training and evaluation functions for Node2Node model.
"""
import math
import sys
from argparse import Namespace
from typing import Iterable, List

import torch
import util.lr_sched as lr_sched
import util.misc as misc
import wandb
from transformers import AutoTokenizer, PreTrainedModel

PRINT_FREQ = 50


@torch.no_grad()
def get_text_embeddings(
    text_model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    texts: List[str],
    device: torch.device,
):
    """
    Return the text embeddings of the given instructions.
    """
    inputs = tokenizer(texts, padding=True, return_tensors="pt")
    inputs = inputs.to(device)
    embeddings = text_model(**inputs).pooler_output
    return embeddings


def get_text_embeddings_grads(
    text_model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    texts: List[str],
    device: torch.device,
):
    """
    Return the text embeddings of the given instructions,
    with gradients.
    """
    inputs = tokenizer(texts, padding=True, return_tensors="pt")
    inputs = inputs.to(device)
    embeddings = text_model(**inputs).pooler_output
    return embeddings


def forward_pass(
    nodes_a: torch.Tensor,
    nodes_b: torch.Tensor,
    text_ab: torch.Tensor,
    labels: torch.Tensor,
    model: torch.nn.Module,
    ae: torch.nn.Module,
    criterion: torch.nn.Module,
    device: torch.device,
    text_model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    args: Namespace = None,
):
    """
    Compute a single forward pass of the model.
    """
    with torch.cuda.amp.autocast(enabled=False):
        labels = labels.to(device, non_blocking=True)
        if args.use_embeds:
            embeds_node_a = nodes_a.to(device, non_blocking=True)
            embeds_node_b = nodes_b.to(device, non_blocking=True)
            if args.ft_bert:
                embeds_text_ab = get_text_embeddings_grads(
                    text_model=text_model,
                    tokenizer=tokenizer,
                    texts=text_ab,
                    device=device,
                )
            else:
                embeds_text_ab = text_ab.to(device, non_blocking=True)
        else:
            nodes_a = nodes_a.to(device, non_blocking=True)
            nodes_b = nodes_b.to(device, non_blocking=True)
            embeds_text_ab = get_text_embeddings(
                text_model=text_model, tokenizer=tokenizer, texts=text_ab, device=device
            )
            with torch.no_grad():
                _, embeds_node_a = ae.encode(nodes_a)
                _, embeds_node_b = ae.encode(nodes_b)

        if args.alt_ae_embeds is None:
            embeds_node_a = embeds_node_a.reshape(embeds_node_a.shape[0], 512, 8)
            embeds_node_b = embeds_node_b.reshape(embeds_node_b.shape[0], 512, 8)

        loss, acc = criterion(
            model, embeds_node_a, embeds_node_b, embeds_text_ab, labels
        )
    return loss, acc


def train_one_epoch(
    data_loader: Iterable,
    model: torch.nn.Module,
    ae: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    text_model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    max_norm: float = 0,
    args: Namespace = None,
    global_rank: int = 0,
):
    """
    Train the model for one epoch.
    """
    model.train(True)
    if args.ft_bert:
        text_model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)

    accum_iter = args.accum_iter
    optimizer.zero_grad()

    for data_iter_step, (nodes_a, nodes_b, text_ab, labels) in enumerate(
        metric_logger.log_every(data_loader, PRINT_FREQ, header)
    ):
        # We use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0 and not (
            args.use_adam or args.plateau_scheduler
        ):
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        # Computing loss
        loss, acc = forward_pass(
            nodes_a=nodes_a,
            nodes_b=nodes_b,
            text_ab=text_ab,
            labels=labels,
            model=model,
            ae=ae,
            criterion=criterion,
            device=device,
            text_model=text_model,
            tokenizer=tokenizer,
            args=args,
        )
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=False,
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(acc=acc)

        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        if global_rank == 0 and (data_iter_step + 1) % accum_iter == 0:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            wandb.log(
                {
                    "epoch_1000x": epoch_1000x,
                    "train_batch_loss": float(loss_value),
                    "train_batch_acc": float(acc),
                    "min_lr": min_lr,
                    "max_lr": max_lr,
                }
            )

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if global_rank == 0:
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": float(metric_logger.loss.global_avg),
                "train_acc": float(metric_logger.acc.global_avg),
                "min_lr": min_lr,
                "max_lr": max_lr,
            }
        )
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.inference_mode()
def evaluate(
    data_loader: Iterable,
    model: torch.nn.Module,
    ae: torch.nn.Module,
    criterion: torch.nn.Module,
    device: torch.device,
    epoch: int,
    text_model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    args: Namespace = None,
    global_rank: int = 0,
):
    """
    Evaluate the model.
    """
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"

    # switch to evaluation mode
    model.eval()

    for nodes_a, nodes_b, prompts_ab, labels in metric_logger.log_every(
        data_loader, PRINT_FREQ, header
    ):
        loss, acc = forward_pass(
            nodes_a=nodes_a,
            nodes_b=nodes_b,
            text_ab=prompts_ab,
            labels=labels,
            model=model,
            ae=ae,
            criterion=criterion,
            device=device,
            text_model=text_model,
            tokenizer=tokenizer,
            args=args,
        )
        loss_value = loss.item()
        metric_logger.update(loss=loss_value)
        metric_logger.update(acc=acc)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("* loss {losses.global_avg:.3f}".format(losses=metric_logger.loss))
    if global_rank == 0:
        wandb.log(
            {
                "epoch": epoch,
                "valid_loss": float(metric_logger.loss.global_avg),
                "valid_acc": float(metric_logger.acc.global_avg),
            }
        )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
