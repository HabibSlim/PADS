from torch.utils.data import Sampler
import numpy as np
import torch
import torch.distributed as dist
import math
from typing import Iterator, List, Optional, TypeVar
from collections import defaultdict

T_co = TypeVar("T_co", covariant=True)


class DistributedGroupBatchSampler(Sampler[T_co]):
    """
    Sampler that combines group-aware batch sampling with distributed sampling capabilities.
    Each batch contains samples from only one group, and data is properly sharded across processes.

    Args:
        group_ids: Array-like of shape [N_SAMPLES] containing group ID for each sample
        batch_size: Maximum number of samples per batch
        num_replicas: Number of processes participating in distributed training
        rank: Rank of the current process
        shuffle: If True, shuffle samples within each group
        seed: Random seed for shuffle reproducibility
        drop_last: If True, drop the last batch if its size would be less than batch_size
    """

    def __init__(
        self,
        group_ids: np.ndarray,
        batch_size: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if len(group_ids) == 0:
            raise ValueError("group_ids cannot be empty")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )

        self.group_ids = np.asarray(group_ids)
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last

        # Group samples by their group IDs
        self.group_to_samples = defaultdict(list)
        unique_groups = set()
        for idx, group_id in enumerate(self.group_ids):
            self.group_to_samples[group_id].append(idx)
            unique_groups.add(group_id)

        if not unique_groups:
            raise ValueError("No valid groups found in group_ids")

        # Convert lists to arrays for faster indexing
        self.group_to_samples = {
            k: np.array(v) for k, v in self.group_to_samples.items()
        }

        # Calculate total number of batches
        total_batches = 0
        for indices in self.group_to_samples.values():
            num_samples = len(indices)
            if num_samples < self.batch_size:
                import warnings

                warnings.warn(
                    f"Found group with {num_samples} samples, less than batch_size ({self.batch_size})"
                )

            if self.drop_last:
                num_group_batches = num_samples // self.batch_size
            else:
                num_group_batches = (
                    num_samples + self.batch_size - 1
                ) // self.batch_size
            total_batches += num_group_batches

        # Ensure even distribution across replicas
        self.num_batches = math.ceil(total_batches / self.num_replicas)
        self.total_size = self.num_batches * self.num_replicas

    def __iter__(self) -> Iterator[List[int]]:
        # Set random seed for this epoch
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Create lists of batches for each group
        all_batches = []

        for group_id, group_indices in self.group_to_samples.items():
            indices = group_indices.copy()

            if self.shuffle:
                # Use PyTorch's RNG for consistent cross-process randomness
                idx_tensor = torch.tensor(indices)
                shuffled = torch.randperm(len(indices), generator=g)
                indices = idx_tensor[shuffled].numpy()

            # Create batches for this group
            num_samples = len(indices)

            # Skip empty groups
            if num_samples == 0:
                continue

            num_full_batches = num_samples // self.batch_size

            # Add full batches
            for batch_idx in range(num_full_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = start_idx + self.batch_size
                batch = indices[start_idx:end_idx].tolist()
                if len(batch) > 0:  # Only add non-empty batches
                    all_batches.append(batch)

            # Handle last incomplete batch
            if not self.drop_last and num_samples % self.batch_size != 0:
                batch = indices[num_full_batches * self.batch_size :].tolist()
                if len(batch) > 0:  # Only add non-empty batches
                    all_batches.append(batch)

        if not all_batches:
            raise RuntimeError("No valid batches created from any group")

        # Verify batch sizes and group homogeneity
        for batch in all_batches:
            if len(batch) > self.batch_size:
                raise RuntimeError(
                    f"Found batch of size {len(batch)}, larger than specified batch_size {self.batch_size}"
                )
            batch_groups = set(self.group_ids[batch])
            if len(batch_groups) > 1:
                raise RuntimeError(f"Batch contains multiple groups: {batch_groups}")

        # Shuffle the order of batches if requested
        if self.shuffle:
            # Use PyTorch's RNG for consistent cross-process randomness
            batch_order = torch.randperm(len(all_batches), generator=g).tolist()
            all_batches = [all_batches[i] for i in batch_order]

        # Handle padding to ensure even distribution
        if len(all_batches) < self.total_size:
            # Find the largest non-empty batch to use for padding
            padding_batch = max(all_batches, key=len)
            while len(all_batches) < self.total_size:
                all_batches.append(padding_batch.copy())
        elif len(all_batches) > self.total_size:
            # If we have too many batches, trim the excess
            all_batches = all_batches[: self.total_size]

        # Select batches for this rank
        rank_batches = all_batches[self.rank : self.total_size : self.num_replicas]
        assert (
            len(rank_batches) == self.num_batches
        ), f"Expected {self.num_batches} batches for rank {self.rank}, got {len(rank_batches)}"

        return iter(rank_batches)

    def __len__(self) -> int:
        return self.num_batches

    def set_epoch(self, epoch: int) -> None:
        """
        Sets the epoch for this sampler. This ensures all replicas
        use a different random ordering for each epoch when shuffle=True.

        Args:
            epoch (int): Epoch number
        """
        self.epoch = epoch
