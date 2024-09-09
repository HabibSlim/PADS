"""
Latents dataset.
"""

import json
import os
import numpy as np
import torch
import torch.nn.functional as F

import random
from torch.utils.data import Dataset, BatchSampler, DataLoader
from collections import defaultdict


class ShapeLatentDataset(Dataset):
    """
    Shape latent dataset.
    """

    PART_CAP = 24

    def __init__(
        self, data_dir, exclude_types=None, cap_parts=True, shuffle_parts=True
    ):
        self.exclude_types = set(exclude_types) if exclude_types else set()
        self.shuffle_parts = shuffle_parts

        file_list = "capped_list.json" if cap_parts else "full_list.json"
        file_list = json.load(open(os.path.join(data_dir, file_list)))
        self.latents_dir = os.path.join(data_dir, "latents")
        self.bbs_dir = os.path.join(data_dir, "bounding_boxes")

        final_list = []
        for f in file_list:
            file_type = "_".join(f.split("_")[2:-1])
            if file_type not in self.exclude_types:
                bb_coords_f = f + "_part_bbs"
                bb_labels_f = f + "_part_labels"
                final_list += [[k + ".npy" for k in [f, bb_coords_f, bb_labels_f]]]
        self.file_list = final_list
        self.file_list.sort()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Unpack file paths
        file_paths = [os.path.join(self.latents_dir, self.file_list[idx][0])]
        file_paths = file_paths + [
            os.path.join(self.bbs_dir, f) for f in self.file_list[idx][1:]
        ]
        latent_f, bb_coords_f, bb_labels_f = file_paths

        # Loading latent and bounding box data
        latent = np.load(latent_f)
        bb_coords = np.load(bb_coords_f)
        bb_labels = np.load(bb_labels_f)

        # Convert numpy array to torch tensor
        latent_tensor = torch.from_numpy(latent).float()
        bb_coords_tensor = torch.from_numpy(bb_coords).float()
        bb_labels_tensor = torch.from_numpy(bb_labels).long()

        # Shuffle the order of parts if self.shuffle is True
        if self.shuffle_parts:
            num_parts = bb_coords_tensor.size(0)
            shuffle_indices = torch.randperm(num_parts)
            bb_coords_tensor = bb_coords_tensor[shuffle_indices]
            bb_labels_tensor = bb_labels_tensor[shuffle_indices]

        # Pad bb coords and labels
        pad_size = self.PART_CAP - bb_coords_tensor.size(0)

        # Pad the tensors
        bb_coords_tensor = F.pad(bb_coords_tensor, (0, 0, 0, 0, 0, pad_size))
        bb_labels_tensor = F.pad(bb_labels_tensor, (0, pad_size), value=-1)

        # Extract metadata from filename
        meta = os.path.basename(latent_f).split(".")[0]

        return (
            latent_tensor,
            bb_coords_tensor,
            bb_labels_tensor,
            meta,
        )


class PairedSampler(BatchSampler):
    """
    Sampling augmented shape pairs.
    """

    def __init__(self, dataset, batch_size, pair_types, shuffle=True, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.pair_types = [t.strip() for t in pair_types.split(",")]
        self.shuffle = shuffle
        self.drop_last = drop_last

        if len(self.pair_types) != 2:
            raise ValueError(
                "pair_types should contain exactly two types separated by a comma"
            )

        # Group indices by ID and type
        self.id_to_indices = defaultdict(lambda: defaultdict(list))
        for idx, (filename, _, _) in enumerate(dataset.file_list):
            parts = filename.split("_")
            id_part = "_".join(parts[:2])
            file_type = "_".join(parts[2:-1])
            self.id_to_indices[id_part][file_type].append(idx)

        # Filter out IDs that don't have both required types
        self.valid_ids = [
            id_part
            for id_part, type_dict in self.id_to_indices.items()
            if all(type in type_dict for type in self.pair_types)
        ]

        self.paired_indices = self._create_paired_indices()

    def _create_paired_indices(self):
        paired_indices = []

        if self.shuffle:
            random.shuffle(self.valid_ids)

        for id_part in self.valid_ids:
            type1, type2 = self.pair_types
            indices1 = self.id_to_indices[id_part][type1]
            indices2 = self.id_to_indices[id_part][type2]

            if type1 == type2:
                # If the same type is requested, ensure we have at least 2 files
                if len(indices1) < 2:
                    continue
                pair = random.sample(indices1, 2)
            else:
                # If different types, take one from each
                pair = [random.choice(indices1), random.choice(indices2)]

            paired_indices.extend(pair)

        return paired_indices

    def __iter__(self):
        # if self.shuffle:
        #    random.shuffle(self.paired_indices)

        batch = []
        for idx in self.paired_indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.paired_indices) // self.batch_size
        else:
            return (len(self.paired_indices) + self.batch_size - 1) // self.batch_size


class PairedShapesLoader:
    """
    Paired shapes loader.
    """

    def __init__(self, dataset, batch_size, pair_types, num_workers, **kwargs):
        self.dataset = dataset
        self.batch_size = batch_size
        self.pair_types = pair_types
        self.num_workers = num_workers
        self.kwargs = kwargs
        self.shuffle = kwargs.get("shuffle", True)
        self.create_dataloader()

    def create_dataloader(self):
        batch_sampler = PairedSampler(
            self.dataset,
            pair_types=self.pair_types,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            **self.kwargs,
        )
        self.iterator = iter(self.dataloader)

    def split_tensor(self, tensor):
        tensor_A = tensor[::2]
        tensor_B = tensor[1::2]
        return tensor_A, tensor_B

    def __iter__(self):
        return self

    def __next__(self):
        try:
            latent_tensor, bb_coords_tensor, bb_labels_tensor, meta = next(
                self.iterator
            )
            latent_A, latent_B = self.split_tensor(latent_tensor)
            bb_coords_A, bb_coords_B = self.split_tensor(bb_coords_tensor)
            bb_labels_A, bb_labels_B = self.split_tensor(bb_labels_tensor)
            meta_A, meta_B = self.split_tensor(meta)
            return (latent_A, bb_coords_A, bb_labels_A, meta_A), (
                latent_B,
                bb_coords_B,
                bb_labels_B,
                meta_B,
            )
        except StopIteration:
            self.create_dataloader()  # Reset the dataloader
            raise StopIteration

    def __len__(self):
        return len(self.dataloader)


class ComposedPairedShapesLoader:
    """
    Composed loader that alternates between batches of multiple shape pair types.
    """

    def __init__(self, dataset, batch_size, pair_types_list, num_workers, **kwargs):
        self.loaders = [
            PairedShapesLoader(dataset, batch_size, pair_types, num_workers, **kwargs)
            for pair_types in pair_types_list
        ]
        self.num_loaders = len(self.loaders)

    def __iter__(self):
        iterators = [iter(loader) for loader in self.loaders]
        while True:
            for iterator in iterators:
                try:
                    yield next(iterator)
                except StopIteration:
                    return

    def __len__(self):
        return max(len(loader) for loader in self.loaders) * self.num_loaders
