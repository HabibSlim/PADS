"""
Main training script for the PartQueries-conditioned diffusion model.
"""

import argparse
import datetime
import json
import os
import time
from pathlib import Path

import torch
import util.misc as misc
import wandb
from schedulefree import AdamWScheduleFree

import models.diffusion as dm
from datasets.latents import ShapeLatentDataset, ComposedPairedShapesLoader, PairType
from datasets.metadata import class_to_hex
from engine_pq_dm import evaluate, train_one_epoch
from util.misc import NativeScalerWithGradNormCount as NativeScaler


def get_args_parser():
    parser = argparse.ArgumentParser("Part-aware PQDM training", add_help=False)
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        help="Experiment name to use",
    )
    parser.add_argument("--epochs", default=800, type=int)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )
    parser.add_argument(
        "--wandb_id",
        type=str,
        default=None,
        help="WandbID of the run to resume from",
    )
    parser.add_argument(
        "--valid_step",
        type=int,
        help="Log validation metrics every N epochs",
    )
    parser.add_argument(
        "--save_every_n",
        type=int,
        help="Save model every N epochs",
    )

    # Model parameters
    parser.add_argument(
        "--layer_depth",
        default=4,
        type=int,
        help="Number of transformer layers in the part encoder",
    )
    parser.add_argument(
        "--part_latents_dim",
        default=128,
        type=int,
        help="Dimension of the individual part latents",
    )
    parser.add_argument(
        "--n_parts",
        default=24,
        type=int,
        help="Number of parts per shape",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="kl_d512_m512_l8_d24_pq",
        help="Part queries encoder to use.",
    )

    # Optimizer parameters
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay (default: 0.01)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        metavar="LR",
        help="Learning rate (absolute lr)",
    )
    parser.add_argument(
        "--blr",
        type=float,
        default=1e-4,
        metavar="LR",
        help="Base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument(
        "--schedule_free",
        action="store_true",
        default=False,
        help="Use schedule-free optimizer",
    )
    parser.add_argument(
        "--transfer_weights",
        action="store_true",
        default=False,
        help="Transfer weights from the base model",
    )
    parser.add_argument(
        "--freeze_dm",
        action="store_true",
        default=False,
        help="Freeze the diffusion model",
    )

    # Gradient clipping
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=5.0,
        help="Gradient clipping value",
    )

    # Loss weights
    parser.add_argument(
        "--rec_weight",
        type=float,
        default=0.7,
        help="Reconstruction loss weight",
    )
    parser.add_argument(
        "--inv_weight",
        type=float,
        default=0.2,
        help="Scale/part drop invariance loss weight",
    )

    # Dataset parameters
    parser.add_argument(
        "--data_path",
        default="",
        type=str,
        help="Dataset path",
    )
    parser.add_argument(
        "--overfit",
        action="store_true",
        default=False,
        help="Overfit to a single batch to debug",
    )
    parser.add_argument(
        "--category_name",
        type=str,
        help="Category name to train on",
    )
    parser.add_argument(
        "--no_part_drop",
        action="store_true",
        default=False,
        help="Disable training on the part drop data",
    )

    # Checkpointing parameters
    parser.add_argument(
        "--output_dir",
        default="./output/",
        help="Path where to save, empty for no saving",
    )
    parser.add_argument("--resume", default="", help="Resume from checkpoint")
    parser.add_argument(
        "--resume_weights",
        action="store_true",
        default=False,
        help="Only resume weights, not optimizer state",
    )
    parser.add_argument(
        "--resume_full_weights",
        action="store_true",
        default=False,
        help="Resume the full model weights with the EDM wrapper",
    )

    parser.add_argument(
        "--device", default="cuda", help="Device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument("--num_workers", default=60, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        default=False,
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument(
        "--debug_run",
        action="store_true",
        help="Make a crash run to check if everything is working",
    )

    # Distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="Number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    return parser


def init_dataloaders(args):
    """
    Instantiate appropriate data loaders.
    """
    # Create your datasets
    filter_n_ids = 1 if args.overfit else None
    class_code = (
        class_to_hex(args.category_name) if args.category_name != "all" else None
    )

    dataset_train = ShapeLatentDataset(
        args.data_path,
        class_code=class_code,
        split="train",
        shuffle_parts=True,
        filter_n_ids=filter_n_ids,
    )

    dataset_val = ShapeLatentDataset(
        args.data_path,
        class_code=class_code,
        split="test",
        shuffle_parts=False,
        filter_n_ids=filter_n_ids,
    )
    args.batch_size = 2 if args.overfit else args.batch_size

    # Defining selected contrastive pairs
    pair_types = [PairType.NO_ROT_PAIR]
    if not args.no_part_drop:
        pair_types += [PairType.PART_DROP]

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # Create the DataLoader using the sampler
    data_loader_train = ComposedPairedShapesLoader(
        dataset_train,
        batch_size=args.batch_size,
        pair_types_list=pair_types,
        num_workers=args.num_workers,
        use_distributed=True,
        num_replicas=num_tasks,
        rank=global_rank,
        seed=args.seed,
        shuffle=True,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    data_loader_val = ComposedPairedShapesLoader(
        dataset_val,
        batch_size=args.batch_size,
        pair_types_list=pair_types,
        num_workers=args.num_workers,
        use_distributed=True,
        num_replicas=num_tasks,
        rank=global_rank,
        seed=args.seed,
        shuffle=False,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # Print size of the dataloaders
    print(
        "Train dataloader size: ",
        len(data_loader_train),
        " batches, for ",
        len(dataset_train),
        " samples.",
    )
    print(
        "Val dataloader size: ",
        len(data_loader_val),
        " batches, for ",
        len(dataset_val),
        " samples.",
    )

    return global_rank, data_loader_train, data_loader_val


def train_model(
    args,
    model,
    model_without_ddp,
    data_loader_train,
    data_loader_val,
    optimizer,
    global_rank,
    loss_scaler,
):
    """
    Train the model.
    """
    print(
        "Start training for [%d] epochs, from epoch [%d]."
        % (args.epochs, args.start_epoch)
    )
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            args=args,
            model=model,
            data_loader=data_loader_train,
            optimizer=optimizer,
            epoch=epoch,
            global_rank=global_rank,
            loss_scaler=loss_scaler,
        )
        if global_rank == 0 and (
            (
                args.output_dir
                and (epoch % args.save_every_n == 0 or epoch + 1 == args.epochs)
            )
            or args.debug_run
        ):
            misc.save_model(
                args=args,
                epoch=epoch,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
            )

        if (epoch % args.valid_step == 0 or epoch + 1 == args.epochs) or args.debug_run:
            data_loader_val.set_epoch(epoch, force_reset=True)
            eval_stats = evaluate(
                args=args,
                model=model,
                data_loader=data_loader_val,
                epoch=epoch,
                global_rank=global_rank,
            )

        if args.debug_run:
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))

    # Finishing wandb run
    if global_rank == 0:
        wandb.finish()


def main(args):
    """
    Main function.
    """
    misc.init_distributed_mode(args)
    device = torch.device(args.device)

    # Fix the seed for reproducibility
    misc.set_all_seeds(args.seed)

    # Instantiate the data loaders
    global_rank, data_loader_train, data_loader_val = init_dataloaders(args)

    if global_rank == 0:
        print("Job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
        print("{}".format(args).replace(", ", ",\n"))
        print("Input args:\n", json.dumps(vars(args), indent=4, sort_keys=True))

    # Load initial model weights
    model = dm.__dict__[args.model_name](
        layer_depth=args.layer_depth,
        n_parts=args.n_parts,
        part_latents_dim=args.part_latents_dim,
    )
    if not args.resume_full_weights:
        if args.transfer_weights:
            base_model = dm.kl_d512_m512_l8_d24()
            misc.load_model(args, base_model)
            model = misc.transfer_weights(base_model, model)
        else:
            misc.load_model(args, model)
    model_without_ddp = model.to(device)

    # Print param count in human readable format
    print("Model param count: ", misc.count_params(model))
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=args.freeze_dm
        )
        model_without_ddp = model.module

    # Compute and display lr/eff lr, bsize/eff bsize
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    print(
        "Base lr: %.2e" % (args.lr * 256 / eff_batch_size)
        + " | Effective lr: %.2e" % args.lr
    )
    print("Accumulate grad iterations: %d" % args.accum_iter)
    print(
        "Batch size per GPU: %d" % args.batch_size
        + " | Effective batch size: %d" % eff_batch_size
    )

    # Define optimizer, loss scaler, etc.
    opt_fn = AdamWScheduleFree if args.schedule_free else torch.optim.AdamW
    if args.resume_full_weights:
        optimizer = opt_fn(
            model_without_ddp.parameters(),
        )
        misc.load_optimizer(args, optimizer)
    else:
        if args.schedule_free:
            optimizer = opt_fn(
                model_without_ddp.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
        else:
            optimizer = opt_fn(
                model_without_ddp.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )

    # Load the model from a checkpoint
    if args.resume_full_weights:
        if "layered" in args.model_name:
            misc.load_model(
                args=args,
                model_without_ddp=model_without_ddp,
            )
        else:
            misc.load_model(
                args=args,
                model_without_ddp=model_without_ddp,
            )

    # Start a new wandb run to track this script
    if not args.eval and global_rank == 0:
        model_config = {
            "layer_depth": args.layer_depth,
            "part_latents_dim": args.part_latents_dim,
            "rec_weight": args.rec_weight,
            "inv_weight": args.inv_weight,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        }

        misc.init_wandb(
            project_name="pq_dm",
            exp_name=args.exp_name,
            model_config=model_config,
            wandb_id=args.wandb_id,
        )

    # Load the model from a checkpoint
    if args.resume_full_weights:
        misc.load_model(
            args=args,
            model_without_ddp=model_without_ddp,
            optimizer=optimizer,
        )

    # Train the model
    start_time = time.time()

    loss_scaler = NativeScaler()
    train_model(
        args=args,
        model=model,
        model_without_ddp=model_without_ddp,
        data_loader_train=data_loader_train,
        data_loader_val=data_loader_val,
        optimizer=optimizer,
        global_rank=global_rank,
        loss_scaler=loss_scaler,
    )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(
        "Done. Total training time for [%d] epochs: [%s]."
        % (args.epochs, total_time_str)
    )


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
