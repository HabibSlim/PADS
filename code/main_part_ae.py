"""
Main training script for the part-aware autoencoder model.
"""

import argparse
import datetime
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import wandb

import util.misc as misc
import models.part_ae as part_ae

from datasets.dummy_datasets import DummyPartDataset, collate_dummy
from datasets.part_occupancies import PartOccupancyDataset, collate
from datasets.grouped_sampler import DistributedGroupBatchSampler
from engine_part_ae import evaluate, train_one_epoch
from util.misc import NativeScalerWithGradNormCount as NativeScaler

torch.set_num_threads(8)


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
    parser.add_argument("--num_workers", default=60, type=int)
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
            data_loader_train.batch_sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            args=args,
            model=model,
            data_loader=data_loader_train,
            optimizer=optimizer,
            epoch=epoch,
            global_rank=global_rank,
            loss_scaler=loss_scaler,
            max_norm=args.clip_grad,
        )
        if global_rank == 0 and (
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
                global_rank=global_rank,
            )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))

    # Finishing wandb run
    if global_rank == 0:
        wandb.finish()


def init_dataloaders(args):
    """
    Initialize the data loaders.
    """
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    if args.debug_run and not args.use_hdf5:
        dataset = DummyPartDataset(
            batch_size=args.batch_size,
            num_samples=512,
            max_parts=4,
            num_points=1024,
            num_occ=2048,
            num_replicas=num_tasks,
        )
        dataset_train, dataset_val = dataset, dataset
    else:
        dataset_train = PartOccupancyDataset(
            hdf5_path=args.data_path,
            split="train",
            num_queries=args.n_query_points,
            num_part_points=args.n_part_points,
            num_replicas=num_tasks,
        )
        dataset_val = PartOccupancyDataset(
            hdf5_path=args.data_path,
            split="val",
            num_queries=args.n_query_points,
            num_part_points=args.n_part_points,
            num_replicas=num_tasks,
        )

    sampler_train = DistributedGroupBatchSampler(
        group_ids=dataset_train.part_counts,
        batch_size=args.batch_size,
        num_replicas=num_tasks,
        rank=global_rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    print("Sampler_train = %s" % str(sampler_train))
    sampler_val = DistributedGroupBatchSampler(
        group_ids=dataset_val.part_counts,
        batch_size=args.batch_size,
        num_replicas=num_tasks,
        rank=global_rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )

    # Initialize the dataloaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_sampler=sampler_train,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        prefetch_factor=2,
        collate_fn=collate,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_sampler=sampler_val,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        prefetch_factor=2,
        collate_fn=collate,
    )

    return global_rank, data_loader_train, data_loader_val


def main(args):
    """
    Main function for training.
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
    model = part_ae.__dict__[args.model_name]()
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

    # Compute and display lr/eff lr, bsize/eff bsize
    init_lr(args)

    # Loading the optimizer and loss scaler
    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr)
    loss_scaler = NativeScaler()

    # Start a new wandb run to track this script
    print("Global rank: ", global_rank)
    print("World size: ", misc.get_world_size())
    if not args.eval and global_rank == 0:
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
            project_name="part_ae_occ",
            exp_name=args.exp_name,
            model_config=model_config,
            wandb_id=args.wandb_id,
        )

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
    args.grid_density = 32 if args.debug_run else 256
    main(args)
