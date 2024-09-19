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


class PairType:
    NO_ROT_PAIR = "rand_no_rot,rand_no_rot"
    PART_DROP = "part_drop,orig"


class ShapeLatentDataset(Dataset):
    """
    Shape latent dataset.
    """

    PART_CAP = 24

    def __init__(
        self,
        data_dir,
        exclude_types=None,
        cap_parts=True,
        shuffle_parts=True,
        class_code=None,
        split=None,
        filter_n_ids=None,
    ):
        exclude_types = set(exclude_types) if exclude_types else set()
        self.shuffle_parts = shuffle_parts

        # Load file list
        file_list = "capped_list.json" if cap_parts else "full_list.json"
        file_list = json.load(open(os.path.join(data_dir, file_list)))
        latents_dir = os.path.join(data_dir, "latents")
        bbs_dir = os.path.join(data_dir, "bounding_boxes")

        # Load the split
        if split is not None:
            split = json.load(open(os.path.join(data_dir, "split_" + split + ".json")))
            split = set(split)

        final_list = []
        for f in file_list:
            if split is not None and f[:6] not in split:
                continue

            file_type = "_".join(f.split("_")[2:-1])

            # Filter by class code
            valid_cls = class_code is None or f.startswith(class_code)
            if not valid_cls:
                continue

            # Filter by file type
            if file_type not in exclude_types:
                bb_coords_f = f + "_part_bbs"
                bb_labels_f = f + "_part_labels"
                final_list += [[k + ".npy" for k in [f, bb_coords_f, bb_labels_f]]]

        # Create a list of file paths
        file_list = final_list
        file_list.sort()
        self.file_list = file_list
        self.file_tuples = []
        for idx in range(len(file_list)):
            # Unpack file paths
            file_paths = [os.path.join(latents_dir, file_list[idx][0])]
            file_paths = file_paths + [
                os.path.join(bbs_dir, f) for f in file_list[idx][1:]
            ]
            latent_f, bb_coords_f, bb_labels_f = file_paths

            # Extract model ID from the filename
            basename = os.path.basename(latent_f)
            model_id = basename.split("_")[0:2][0] + basename.split("_")[0:2][1]
            model_id = int(model_id, 16)
            self.file_tuples += [(latent_f, bb_coords_f, bb_labels_f, model_id)]

        if filter_n_ids is not None:
            # Only keep the sampels corresponding to N=filter_n_ids distinct model IDs
            unique_model_ids = list(set(tup[3] for tup in self.file_tuples))
            random.shuffle(unique_model_ids)
            selected_model_ids = set(unique_model_ids[:filter_n_ids])

            self.file_tuples = [
                tuple for tuple in self.file_tuples if tuple[3] in selected_model_ids
            ]
            filtered_file_paths = [tuple[0] for tuple in self.file_tuples]
            self.file_list = [
                sublist
                for sublist in self.file_list
                if os.path.join(latents_dir, sublist[0]) in filtered_file_paths
            ]

        self.rng = torch.Generator()

    def __len__(self):
        return len(self.file_tuples)

    def __getitem__(self, idx):
        # Unpack file paths
        latent_f, bb_coords_f, bb_labels_f, model_id = self.file_tuples[idx]

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
            self.rng.manual_seed(model_id)

            num_parts = bb_coords_tensor.size(0)
            shuffle_indices = torch.randperm(num_parts, generator=self.rng)
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
        pair_types = [t.strip() for t in pair_types.split(",")]
        if len(pair_types) != 2:
            raise ValueError(
                "pair_types should contain exactly two types separated by a comma"
            )

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Group indices by ID and type
        id_to_indices = defaultdict(lambda: defaultdict(list))
        for idx, (filename, _, _) in enumerate(dataset.file_list):
            parts = filename.split("_")
            id_part = "_".join(parts[:2])
            file_type = "_".join(parts[2:-1])
            id_to_indices[id_part][file_type].append(idx)

        # Filter out IDs that don't have both required types
        valid_ids = [
            id_part
            for id_part, type_dict in id_to_indices.items()
            if all(p_type in type_dict for p_type in pair_types)
        ]

        self.paired_indices = self._create_paired_indices(
            id_to_indices, valid_ids, pair_types
        )

    def _create_paired_indices(self, id_to_indices, valid_ids, pair_types):
        paired_indices = []

        if self.shuffle:
            random.shuffle(valid_ids)

        for id_part in valid_ids:
            type1, type2 = pair_types
            indices1 = id_to_indices[id_part][type1]
            indices2 = id_to_indices[id_part][type2]

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
        batch = []
        for idx in self.paired_indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

    def __len__(self):
        return len(self.paired_indices) * 2 // self.batch_size


class DistributedPairedSampler(BatchSampler):
    def __init__(
        self,
        dataset,
        batch_size,
        pair_types,
        num_replicas=None,
        rank=None,
        seed=0,
        shuffle=True,
        drop_last=False,
    ):
        pair_types = [t.strip() for t in pair_types.split(",")]
        if len(pair_types) != 2:
            raise ValueError(
                "pair_types should contain exactly two types separated by a comma"
            )

        self.dataset = dataset
        self.batch_size = batch_size
        self.epoch = 0
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Create paired indices
        self.paired_indices = self._create_paired_indices(dataset.file_list, pair_types)
        self.num_samples = len(self.paired_indices) // self.num_replicas

        # Create RNG
        self.rng = torch.Generator()
        self.rng.manual_seed(self.seed + self.epoch)
        self.indices = None

    def _create_paired_indices(self, file_list, pair_types):
        """
        Initialize a list of paired indices.
        """

        # Group indices by ID and type
        id_to_indices = defaultdict(lambda: defaultdict(list))
        for idx, (filename, _, _) in enumerate(file_list):
            parts = filename.split("_")
            id_part = "_".join(parts[:2])
            file_type = "_".join(parts[2:-1])
            id_to_indices[id_part][file_type].append(idx)

        # Filter out IDs that don't have both required types
        valid_ids = [
            id_part
            for id_part, type_dict in id_to_indices.items()
            if all(type in type_dict for type in pair_types)
        ]

        paired_indices = []

        for id_part in valid_ids:
            type1, type2 = pair_types
            indices1 = id_to_indices[id_part][type1]
            indices2 = id_to_indices[id_part][type2]

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

    def sample_indices(self):
        """
        Sample indices for the current epoch.
        """
        # Deterministically shuffle based on epoch and seed
        if self.shuffle:
            n = len(self.paired_indices)

            # Generate a permutation for N/2 pairs
            pair_perm = torch.randperm(n // 2, generator=self.rng).tolist()

            # Use the permutation to reindex the paired list
            indices = [j for i in pair_perm for j in (2 * i, 2 * i + 1)]
        else:
            indices = list(range(len(self.paired_indices)))

        # Subsample while preserving pairs
        n_pairs = len(indices) // 2
        pair_indices = list(range(n_pairs))
        subsampled_pair_indices = pair_indices[self.rank : n_pairs : self.num_replicas]

        self.indices = [
            idx
            for pair_idx in subsampled_pair_indices
            for idx in indices[2 * pair_idx : 2 * pair_idx + 2]
        ]

    def __iter__(self):
        if self.indices is None:
            self.sample_indices()

        # Create batches
        batches = []
        batch = []
        for idx in self.indices:
            batch.append(self.paired_indices[idx])
            if len(batch) == self.batch_size:
                batches.append(batch)
                batch = []

        return iter(batches)

    def __len__(self):
        num_samples = len(self.paired_indices)
        return num_samples // self.batch_size

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.rng.manual_seed(self.seed + self.epoch)
        self.sample_indices()


class PairedShapesLoader:
    """
    Paired shapes loader.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        pair_types,
        num_workers,
        shuffle,
        use_distributed=False,
        num_replicas=None,
        rank=None,
        **kwargs,
    ):
        # Filter out keys from kwargs that are not DataLoader arguments
        valid_keys = set(DataLoader.__init__.__code__.co_varnames)
        kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
        self.kwargs = kwargs
        self.dataset = dataset
        self.batch_size = batch_size
        self.pair_types = pair_types
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.use_distributed = use_distributed
        self.num_replicas = num_replicas
        self.rank = rank
        self.create_dataloader()

    def create_dataloader(self):
        if self.use_distributed:
            batch_sampler = DistributedPairedSampler(
                self.dataset,
                self.batch_size,
                self.pair_types,
                shuffle=self.shuffle,
                num_replicas=self.num_replicas,
                rank=self.rank,
            )
        else:
            batch_sampler = PairedSampler(
                self.dataset,
                pair_types=self.pair_types,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
            )
        self.sampler = batch_sampler
        self.dataloader = DataLoader(
            self.dataset,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
        )
        self.iterator = iter(self.dataloader)

    def set_epoch(self, epoch):
        if self.use_distributed:
            self.sampler.set_epoch(epoch)
            self.iterator = iter(self.dataloader)
        else:
            raise ValueError("set_epoch is only supported in distributed mode")

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
            self.iterator = iter(self.dataloader)
            raise StopIteration

    def __len__(self):
        return len(self.dataloader)


class ComposedPairedShapesLoader:
    """
    Composed loader that alternates between batches of multiple shape pair types.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        pair_types_list,
        num_workers,
        shuffle=False,
        use_distributed=False,
        num_replicas=None,
        rank=None,
        reset_every=100,
        **kwargs,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.pair_types_list = pair_types_list
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.use_distributed = use_distributed
        self.num_replicas = num_replicas
        self.rank = rank
        self.kwargs = kwargs
        self.reset_every = reset_every
        self.loaders = None

    def create_loaders(self):
        self.loaders = [
            (
                pair_types,
                PairedShapesLoader(
                    self.dataset,
                    self.batch_size,
                    pair_types,
                    self.num_workers,
                    shuffle=self.shuffle,
                    use_distributed=self.use_distributed,
                    num_replicas=self.num_replicas,
                    rank=self.rank,
                    **self.kwargs,
                ),
            )
            for pair_types in self.pair_types_list
        ]
        self.num_loaders = len(self.loaders)

    def set_epoch(self, epoch, force_reset=False):
        if epoch % self.reset_every == 0 or force_reset:
            self.create_loaders()
        for _, loader in self.loaders:
            loader.set_epoch(epoch)

    def __iter__(self):
        if self.loaders is None:
            self.create_loaders()
        iterators = [(pair_types, iter(loader)) for pair_types, loader in self.loaders]
        while True:
            for pair_types, iterator in iterators:
                try:
                    yield pair_types, *next(iterator)
                except StopIteration:
                    return

    def __len__(self):
        if self.loaders is None:
            self.create_loaders()
        return max(len(loader) for _, loader in self.loaders)
