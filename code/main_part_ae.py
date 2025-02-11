"""
Main training script for the part-aware autoencoder model.
"""

import argparse
import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import wandb

import util.misc as misc
import models.part_ae as part_ae

from datasets.dummy_datasets import DummyPartDataset, collate_dummy
from datasets.part_occupancies import (
    PartOccupancyDataset,
    ShardedPartOccupancyDataset,
    collate,
)
from datasets.grouped_sampler import (
    DistributedGroupBatchSampler,
    ShardedGroupBatchSampler,
)
from engine_part_ae import evaluate, train_one_epoch
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.monitors import ActivationMonitor, GradientMonitor

# torch.set_num_threads(8)
# torch.set_float32_matmul_precision("low")

sys.stdout.reconfigure(line_buffering=True)


def get_args_parser():
    """
    Argument parser for the part-aware autoencoder model.
    """
    parser = argparse.ArgumentParser("Part-aware autoencoder model", add_help=False)

    # Wandb parameters
    parser.add_argument(
        "--wandb_id",
        type=str,
        default=None,
        help="WandbID of the run to resume from",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        help="Experiment name to use",
    )

    # Training
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument(
        "--accum_iter",
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )
    parser.add_argument(
        "--debug_run",
        action="store_true",
        default=False,
        help="Debug run with minimal number of steps and a dummy dataset",
    )
    parser.add_argument(
        "--gradient_monitoring",
        action="store_true",
        default=False,
        help="Use gradient monitoring for debugging",
    )
    parser.add_argument(
        "--use_hdf5",
        action="store_true",
        default=False,
        help="Debug run using the HDF5 dataset",
    )
    parser.add_argument(
        "--n_part_points",
        type=int,
        default=1024,
        help="Number of sampled points for each part",
    )
    parser.add_argument(
        "--n_query_points",
        type=int,
        default=2048,
        help="Number of query points for occupancy prediction",
    )
    parser.add_argument(
        "--grid_density",
        type=int,
        default=64,
        help="Number of sampled points for each part",
    )

    # Start/End epoch
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--epochs", default=800, type=int)

    # Model parameters
    parser.add_argument(
        "--model_name",
        type=str,
        help="Part-aware autoencoder model to use",
    )

    # Loss function weights
    parser.add_argument("--kl_weight", type=float, help="Weight for KL loss term")
    parser.add_argument(
        "--near_weights",
        type=float,
        nargs="+",
        help="Weights for near-surface point clouds",
    )
    parser.add_argument(
        "--default_near_weights",
        type=float,
        help="Default weight for near-surface loss when near_weights is None",
    )
    parser.add_argument("--vol_weight", type=float, help="Weight for volume loss term")

    # Optimizer parameters
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=5.0,
        help="Gradient clipping value",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        metavar="LR",
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--blr",
        type=float,
        default=1e-4,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument(
        "--layer_decay",
        type=float,
        default=0.75,
        help="layer-wise lr decay from ELECTRA/BEiT",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0",
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=40, metavar="N", help="epochs to warmup LR"
    )

    # Dataset parameters
    parser.add_argument(
        "--data_path",
        default="",
        type=str,
        help="Path to the HDF5 dataset",
    )

    # Checkpointing parameters
    parser.add_argument("--resume", default="", help="Resume from checkpoint")
    parser.add_argument(
        "--log_dir", default="./output/", help="path where to tensorboard log"
    )
    parser.add_argument(
        "--output_dir",
        help="Path to save model checkpoints",
    )

    # Log/checkpointing frequency
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

    # CUDA parameters
    parser.add_argument(
        "--device", default="cuda", help="Device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--dist_eval",
        action="store_true",
        default=False,
        help="Enabling distributed evaluation (recommended during training for faster monitor",
    )
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=False)

    # Distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    return parser


def init_lr(args):
    """
    Initialize effective and base learning rates based on input parameters.
    """
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    print("Accumulate grad iterations: %d" % args.accum_iter)
    print(
        "Batch size per GPU: %d" % args.batch_size
        + " | Effective batch size: %d" % eff_batch_size
    )

    print("Base lr: %.2e" % (args.blr))
    print("Effective lr: %.2e" % args.lr)


def train_model(
    args,
    model,
    model_without_ddp,
    data_loader_train,
    data_loader_val,
    optimizer,
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
            data_loader_train.batch_sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            args=args,
            model=model,
            data_loader=data_loader_train,
            optimizer=optimizer,
            epoch=epoch,
            loss_scaler=loss_scaler,
            max_norm=args.clip_grad,
        )
        if args.global_rank == 0 and (
            args.output_dir
            and (epoch % args.save_every_n == 0 or epoch + 1 == args.epochs)
        ):
            misc.save_model(
                args=args,
                epoch=epoch,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
            )

        if epoch % args.valid_step == 0 or epoch + 1 == args.epochs:
            data_loader_val.batch_sampler.set_epoch(epoch)
            eval_stats = evaluate(
                args=args,
                model=model,
                data_loader=data_loader_val,
                epoch=epoch,
            )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))

    # Finishing wandb run
    if args.global_rank == 0:
        wandb.finish()


def init_dataloaders(args):
    """
    Initialize the data loaders.
    """
    if args.debug_run and not args.use_hdf5:
        dataset = DummyPartDataset(
            num_samples=80000,
            num_part_points=args.n_part_points,
            num_queries=args.n_query_points,
            num_replicas=args.num_tasks,
            max_parts=25,
            full_parts=True,
        )
        print(
            "Using DummyPartDataset with %d samples and %d parts"
            % (len(dataset), dataset.max_parts)
        )
    else:
        dataset = ShardedPartOccupancyDataset(
            hdf5_path=args.data_path,
            rank=args.global_rank,
            world_size=args.num_tasks,
            num_queries=args.n_query_points,
            num_part_points=args.n_part_points,
        )
        print(
            "Using PartOccupancyDataset with %d train samples"
            % (len(dataset) * args.num_tasks)
        )

    # Initialize the samplers
    sampler_train = ShardedGroupBatchSampler(
        group_ids=dataset.part_counts,
        split="train",
        batch_size=args.batch_size,
        num_replicas=args.num_tasks,
        rank=args.global_rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    sampler_val = ShardedGroupBatchSampler(
        group_ids=dataset.part_counts,
        split="val",
        batch_size=2,
        num_replicas=args.num_tasks,
        rank=args.global_rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )

    # Initialize the dataloaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler_train,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        prefetch_factor=2,
        collate_fn=collate,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler_val,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        prefetch_factor=2,
        collate_fn=collate,
    )

    return data_loader_train, data_loader_val


def main(args):
    """
    Main function for training.
    """
    misc.init_distributed_mode(args)
    device = torch.device(args.device)

    # Fetching distributed learning info
    args.num_tasks = misc.get_world_size()
    args.global_rank = misc.get_rank()

    # Fix the seed for reproducibility
    misc.set_all_seeds(args.seed)

    # Start a new wandb run to track this script
    print("Global rank: ", args.global_rank)
    print("World size: ", args.num_tasks)
    if not args.eval and args.global_rank == 0:
        model_config = {
            "model_name": args.model_name,
            "batch_size": args.batch_size,
            "accum_iter": args.accum_iter,
            "clip_grad": args.clip_grad,
            "weight_decay": args.weight_decay,
            "lr": args.lr,
            "blr": args.blr,
            "layer_decay": args.layer_decay,
            "min_lr": args.min_lr,
            "warmup_epochs": args.warmup_epochs,
        }
        misc.init_wandb(
            project_name="part_autoencoder",
            exp_name=args.exp_name,
            model_config=model_config,
            wandb_id=args.wandb_id,
        )

    # Instantiate the data loaders
    data_loader_train, data_loader_val = init_dataloaders(args)

    if args.global_rank == 0:
        print("Job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
        print("{}".format(args).replace(", ", ",\n"))
        print("Input args:\n", json.dumps(vars(args), indent=4, sort_keys=True))

    # Load initial model weights
    model = part_ae.__dict__[args.model_name](args)
    model_without_ddp = model.to(device)

    # Print param count in human readable format
    print("Model param count: ", misc.count_params(model))
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=False,
        )
        model_without_ddp = model.module

    # Instantiate gradient monitor
    if args.gradient_monitoring:
        # monitor = GradientMonitor(
        #     model.module,
        #     threshold=2.0,
        #     norm_type="L2",
        #     break_on_nan=True,
        # )
        # Initialize monitor with appropriate settings
        monitor = ActivationMonitor(
            model,
            threshold=2.0,
            norm_type="L2",
            break_on_nan=True,
            log_level=logging.INFO,  # Adjust based on needs
        )

        args.monitor = monitor

    # Compile the model
    # print("Compiling the model...")
    # model = torch.compile(model)
    # print("Compilation done.")

    # Compute and display lr/eff lr, bsize/eff bsize
    init_lr(args)

    # Loading the optimizer and loss scaler
    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr)
    loss_scaler = NativeScaler()

    # Load the model from a checkpoint
    if args.resume:
        model_without_ddp = misc.load_model(
            args=args,
            model_without_ddp=model_without_ddp,
            optimizer=optimizer,
        )

    # Train the model
    start_time = time.time()

    train_model(
        args=args,
        model=model,
        model_without_ddp=model_without_ddp,
        data_loader_train=data_loader_train,
        data_loader_val=data_loader_val,
        optimizer=optimizer,
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
