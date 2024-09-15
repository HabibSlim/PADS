"""
Main training script for text-conditioned node to node diffusion models.
"""

import argparse
import datetime
import json
import os
import time
from pathlib import Path

import models.autoencoders as autoencoders
import models.diffusion as diffusion
import models.mlp_mapper as mlp_mapper
import models.transformer_mapper as transformer_mapper
import models.listeners as listeners
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import util.misc as misc
import wandb
import engine_node2node
import engine_listener
from losses.edmloss_n2n import Node2NodeLoss
from losses.mapper_loss import MapperLoss, MapperLossDirect
from losses.listener_loss import ListenerLoss
from transformers import AutoTokenizer, BertModel, BertTokenizer, CLIPTextModel
from util.datasets import build_shape_surface_occupancy_dataset
from util.misc import NativeScalerWithGradNormCount as NativeScaler

MODEL_MAP = {
    "kl": diffusion,
    "ae": autoencoders,
    "mlp": mlp_mapper,
    "nrl": listeners,
    "tfm": transformer_mapper,
}


def get_args_parser():
    """
    Get arguments parser.
    """
    parser = argparse.ArgumentParser(
        "Training text-conditioned models.", add_help=False
    )
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
    parser.add_argument(
        "--debug_mode",
        action="store_true",
        default=False,
        help="Run in debug mode",
    )
    parser.add_argument(
        "--debug_with_forward",
        action="store_true",
        default=False,
        help="Run in debug mode, also run forward passes",
    )
    parser.add_argument(
        "--plateau_scheduler",
        action="store_true",
        default=False,
        help="Reduce LR on plateau",
    )
    parser.add_argument(
        "--text_model_name",
        type=str,
        help="Text model name to use",
    )
    parser.add_argument(
        "--wandb_id",
        type=str,
        default=None,
        help="WandbID of the run to resume from",
    )
    parser.add_argument("--epochs", default=800, type=int)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )
    parser.add_argument(
        "--valid_step",
        default=5,
        type=int,
        help="Log validation metrics every N epochs",
    )
    parser.add_argument(
        "--save_every_n",
        default=10,
        type=int,
        help="Save model every N epochs",
    )

    # Model parameters
    parser.add_argument(
        "--model",
        default="kl_d512_m512_l8_edm",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    parser.add_argument(
        "--ae",
        default="kl_d512_m512_l8",
        type=str,
        metavar="MODEL",
        help="Name of autoencoder",
    )
    parser.add_argument("--ae_pth", help="Autoencoder checkpoint")
    parser.add_argument(
        "--ft_bert",
        action="store_true",
        default=False,
        help="Also fine-tune the BERT model",
    )

    # Optimizer parameters
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
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
        default=1e-8,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0",
    )
    parser.add_argument(
        "--intensity_loss",
        action="store_true",
        default=False,
        help="Contrastive edit intensity loss using ground-truth labels.",
    )
    parser.add_argument(
        "--use_adam",
        action="store_true",
        default=False,
        help="Use Adam instead of AdamW.",
    )
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="Start epoch"
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=40, metavar="N", help="epochs to warmup LR"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--n_replicas",
        default=16,
        type=int,
        help="Number of replicas to use for each sample in the dataset",
    )

    # Dataset parameters
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["graphedits"],
        help="Dataset name",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Dataset path",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        help="dataset type",
    )
    parser.add_argument(
        "--max_edge_level",
        default=1,
        type=int,
        help="Maximum edge level to use",
    )
    parser.add_argument(
        "--point_cloud_size",
        default=2048,
        type=int,
        help="Number of points to sample",
    )
    parser.add_argument(
        "--use_embeds",
        action="store_true",
        default=False,
        help="Use precomputed embeddings",
    )
    parser.add_argument(
        "--alt_ae_embeds",
        type=str,
        default=None,
        help="Alternative autoencoder embeddings to use",
    )
    parser.add_argument(
        "--fetch_keys",
        action="store_true",
        default=False,
        help="Fetch node keys in the dataloader",
    )

    # Checkpointing parameters
    parser.add_argument(
        "--output_dir",
        default="./output/",
        help="Path for saving weights/logs",
    )
    parser.add_argument(
        "--log_dir", default="./output/", help="Path where to tensorboard log"
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
        "--resume_mismatch",
        action="store_true",
        default=False,
        help="Resume the full model weights with the EDM wrapper and deal with incompatible keys",
    )
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--dist_eval",
        action="store_true",
        default=False,
        help="Enabling distributed evaluation (recommended during training for faster monitor",
    )

    # Hardware parameters
    parser.add_argument(
        "--device", default="cuda", help="Device to use for training / testing"
    )
    parser.add_argument("--num_workers", default=60, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

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


def main(args):
    """
    Main training routine.
    """
    args.use_clip = "clip" in args.text_model_name

    args.is_diff = args.model.startswith("kl")
    args.is_mlp = args.model.startswith("mlp")
    args.is_nrl_listener = args.model.startswith("nrl")

    args.is_direct = "_direct" in args.model

    if args.is_nrl_listener:
        args.dataset += "_nrl"

    misc.init_distributed_mode(args)
    device = torch.device(args.device)

    # Fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = build_shape_surface_occupancy_dataset("train", args=args)
    dataset_val = build_shape_surface_occupancy_dataset("val", args=args)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print(
                    "Warning: Enabling distributed evaluation"
                    "with an eval dataset not divisible by process number. "
                    "This will slightly alter validation results as"
                    " extra duplicate entries are added to achieve "
                    "equal num of samples per-process."
                )
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0:
        print("Input args:\n", json.dumps(vars(args), indent=4, sort_keys=True))
        print("Job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
        os.makedirs(args.log_dir, exist_ok=True)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # Print dataloader statistics
    if global_rank == 0:
        print("|Train| size = [%d]" % (len(dataset_train) / num_tasks))
        print("|Valid| size = [%d]" % len(dataset_val))

    if not args.use_embeds:
        print("Loading autoencoder %s" % args.ae_pth)
        ae = autoencoders.__dict__[args.ae]()
        ae.eval()
        ae.load_state_dict(torch.load(args.ae_pth, map_location="cpu")["model"])
        ae.to(device)
    else:
        ae = None

    model_module = args.model.split("_")[0]
    model_module = MODEL_MAP[model_module]
    model = model_module.__dict__[args.model](use_linear_proj=not args.use_clip)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Initialize text CLIP model
    if args.use_embeds and not args.ft_bert:
        text_model, tokenizer = None, None
    else:
        if args.use_clip:
            # Instantiate tokenizer + CLIP model
            tokenizer = AutoTokenizer.from_pretrained(args.text_model_name)
            text_model = CLIPTextModel.from_pretrained(args.text_model_name).to(device)
            print("Loaded CLIP model.")
        else:
            # Instantiate tokenizer
            tokenizer = BertTokenizer.from_pretrained(args.text_model_name)
            # Instantiate BERT model and create linear projection layer
            text_model = BertModel.from_pretrained(args.text_model_name).to(device)
            print(
                "Loaded BERT model in [%s] mode." % "fine-tuning"
                if args.ft_bert
                else "eval"
            )

        # Wrap the text model in a DistributedDataParallel module
        if args.distributed:
            text_model_no_ddp = text_model
            text_model = torch.nn.parallel.DistributedDataParallel(
                text_model, device_ids=[args.gpu]
            )

    print("Model = %s" % str(model_without_ddp))
    print("Params (M): %.2f" % (n_parameters / 1.0e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    # Start a new wandb run to track this script
    if not args.eval and global_rank == 0:
        model_config = {
            "batch_size": args.batch_size,
            "eff_batch_size": eff_batch_size,
            "epochs": args.epochs,
            "accum_iter": args.accum_iter,
            "model": args.model,
            "clip_grad": args.clip_grad,
            "weight_decay": args.weight_decay,
            "lr": args.lr,
            "min_lr": args.min_lr,
            "base_blr": args.blr,
            "layer_decay": args.layer_decay,
            "max_edge_level": args.max_edge_level,
            "resume": args.resume,
            "resume_full_weights": args.resume_full_weights,
            "start_epoch": args.start_epoch,
            "eval": args.eval,
            "dist_eval": args.dist_eval,
        }

        misc.init_wandb(
            project_name="shape2vecset",
            exp_name=args.exp_name,
            model_config=model_config,
            wandb_id=args.wandb_id,
        )

    if args.distributed:
        find_used = args.is_diff and not (
            args.resume_mismatch or args.resume_full_weights
        )
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=find_used
        )
        model_without_ddp = model.module

    if args.use_adam:
        optimizer = torch.optim.Adam(
            model_without_ddp.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr)
    if args.plateau_scheduler:
        lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.8, patience=2, verbose=True
        )
    loss_scaler = NativeScaler()

    # Initializing node to node loss
    if args.is_direct:
        criterion = MapperLossDirect()
    else:
        if args.is_mlp:
            criterion = MapperLoss()
        elif args.is_nrl_listener:
            criterion = ListenerLoss()
        else:
            criterion = Node2NodeLoss()

    if args.resume_full_weights:
        misc.load_model(
            args=args,
            model_without_ddp=model_without_ddp,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
        )
        print("Loaded full EDM wrapper + DM weights from checkpoint.")
    elif args.resume_mismatch:
        if "no_ca" in args.model:
            misc.load_model_mismatch_no_ca(
                args=args,
                model_without_ddp=model_without_ddp.edm_model,
            )
        else:
            misc.load_model_mismatch(
                args=args,
                model_without_ddp=model_without_ddp.edm_model,
            )
        print("Loaded EDM model weights selectively.")
    elif args.resume_weights:
        misc.load_model_only(
            args=args,
            model_without_ddp=model_without_ddp.edm_model,
        )
        print("Loaded EDM model weights from checkpoint.")
        # Remove the category embedding layer
        print(
            "Embedding shape:", model_without_ddp.edm_model.embed_ab.weight.data.shape
        )
        model_without_ddp.edm_model.embed_ab = nn.Identity()

    if args.eval:
        raise NotImplementedError()

    # Compile the "train_one_epoch" function
    if not args.debug_mode or (args.debug_mode and args.debug_with_forward):
        if args.is_nrl_listener:
            train_func = engine_listener.train_one_epoch
            eval_func = engine_listener.evaluate
        else:
            train_func = engine_node2node.train_one_epoch
            eval_func = engine_node2node.evaluate
    else:
        train_func = lambda **kwargs: None
        eval_func = lambda **kwargs: {"loss": 0.0}

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    if args.debug_mode and not args.resume:
        epoch_range = range(1)
    else:
        epoch_range = range(args.start_epoch, args.epochs)

    for epoch in epoch_range:
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_func(
            data_loader=data_loader_train,
            model=model,
            ae=ae,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            text_model=text_model,
            tokenizer=tokenizer,
            max_norm=args.clip_grad,
            args=args,
            global_rank=global_rank,
        )
        if (
            global_rank == 0
            and args.output_dir
            and (epoch % args.save_every_n == 0 or epoch + 1 == args.epochs)
        ):
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )
            # Also save the BERT model
            if args.ft_bert:
                misc.save_bert(
                    args=args,
                    epoch=epoch,
                    text_model=text_model_no_ddp,
                )

        if epoch % args.valid_step == 0 or epoch + 1 == args.epochs:
            val_stats = eval_func(
                data_loader=data_loader_val,
                model=model,
                ae=ae,
                criterion=criterion,
                device=device,
                epoch=epoch,
                text_model=text_model,
                tokenizer=tokenizer,
                args=args,
                global_rank=global_rank,
            )
            print(
                f"loss on {len(dataset_val)} validation shape pairs: {val_stats['loss']:.3f}"
            )
            if args.plateau_scheduler:
                lr_sched.step(val_stats["loss"])

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))

    # Finishing wandb run
    if global_rank == 0:
        wandb.finish()


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
