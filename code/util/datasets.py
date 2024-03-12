"""
Dataset utility functions.
"""
import torch
from util.graph_edits import (
    GraphEdits,
    GraphEditsEmbeds,
    GraphEditsEmbedsNRL,
    GraphEditsEmbedsChained,
)


class AxisScaling(object):
    def __init__(self, interval=(0.75, 1.25), jitter=True):
        assert isinstance(interval, tuple)
        self.interval = interval
        self.jitter = jitter

    def __call__(self, surface, point=None):
        scaling = torch.rand(1, 3) * 0.5 + 0.75
        surface = surface * scaling
        if point is not None:
            point = point * scaling

        scale = (1 / torch.abs(surface).max().item()) * 0.999999
        surface *= scale
        if point is not None:
            point *= scale

        if self.jitter:
            surface += 0.005 * torch.randn_like(surface)
            surface.clamp_(min=-1, max=1)

        if point is not None:
            return surface, point
        else:
            return surface


def create_dataset(split, args, sampling, transform, num_samples):
    """
    Create a dataset split instance.
    """
    dataset_modules = {
        "graphedits": GraphEdits,
        "graphedits_nrl": None,
        "graphedits_chained": None,
    }
    dataset_modules_embeds = {
        "graphedits": GraphEditsEmbeds,
        "graphedits_nrl": GraphEditsEmbedsNRL,
        "graphedits_chained": GraphEditsEmbedsChained,
    }

    if hasattr(args, "n_replicas"):
        n_replicas = args.n_replicas if split == "train" else 1
    else:
        n_replicas = 1

    try:
        if args.use_embeds:
            if args.dataset == "graphedits_chained":
                return dataset_modules_embeds[args.dataset](
                    alt_ae_embeds=args.alt_ae_embeds,
                    chain_length=args.chain_length,
                    dataset_folder=args.data_path,
                    dataset_type=args.data_type,
                    replica=n_replicas,
                    split=split,
                    transform=transform,
                    fetch_keys=args.fetch_keys,
                    fetch_intensity=args.intensity_loss,
                    fetch_text_prompts=args.ft_bert,
                )
            return dataset_modules_embeds[args.dataset](
                alt_ae_embeds=args.alt_ae_embeds,
                dataset_folder=args.data_path,
                dataset_type=args.data_type,
                replica=n_replicas,
                split=split,
                transform=transform,
                fetch_keys=args.fetch_keys,
                fetch_intensity=args.intensity_loss,
                fetch_text_prompts=args.ft_bert,
            )
        else:
            return dataset_modules[args.dataset](
                dataset_folder=args.data_path,
                dataset_type=args.data_type,
                replica=n_replicas,
                pc_size=args.point_cloud_size,
                split=split,
                transform=transform,
                sampling=sampling,
                num_samples=num_samples,
                max_edge_level=args.max_edge_level,
                get_voxels=args.get_voxels,
                fetch_keys=args.fetch_keys,
                fetch_intensity=args.intensity_loss,
            )
    # catch ANY exception and print
    except Exception as e:
        print(e)
        print(e)
        print(e)
        exit(0)


def build_shape_surface_occupancy_dataset(split, args, transform=None):
    if split == "train":
        return create_dataset(
            split="train",
            args=args,
            sampling=True,
            transform=transform,
            num_samples=1024,
        )
    if split == "train_sequential":
        return create_dataset(
            split="train",
            args=args,
            sampling=False,
            transform=transform,
            num_samples=1024,
        )
    else:
        return create_dataset(
            split="val",
            args=args,
            sampling=False,
            transform=transform,
            num_samples=1024,
        )
