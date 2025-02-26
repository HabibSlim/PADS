import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from util.transforms import random_transformation_matrix


class PartOccupancyDataset(Dataset):
    """
    Dataset for part-based occupancy prediction with part dropping.
    """

    MAX_PART_DROP = 16
    N_SAMPLING_FNS = 3

    def __init__(
        self,
        rank,
        hdf5_path,
        num_queries=2048,
        num_part_points=2048,
        random_transform=False,
        rot_angle=0.0,
        max_scale=0.0,
    ):
        """
        Initialize the dataset by loading HDF5 matrices into memory.

        Args:
            hdf5_path: Path to the HDF5 file containing the dataset
            num_queries: Number of query points to sample (if None, uses all points)
            num_part_points: Number of points per part (if None, uses all points)
            random_transform: Whether to apply random transformations
            rot_angle: Maximum rotation angle (applied to all axes) in degrees
            max_scale: Maximum scale factor (applied to all axes)
        """

        # Initialize base attributes
        self.rank = rank
        self.num_queries = num_queries
        self.num_part_points = num_part_points

        # Store transformation parameters
        self.random_transform = random_transform
        self.rot_angle = np.radians(rot_angle)
        self.max_scale = max_scale

        # Load and validate HDF5 data
        print("Loading HDF5 data to memory...")
        print()

        with h5py.File(hdf5_path, "r") as f:
            # Load data into memory
            self.model_ids = f["model_ids"][:].astype("U")
            self.part_slices = f["part_slices"][:]
            self.part_drops = f["part_drops"][:]
            self.part_points = f["part_points_matrix"][:]
            self.part_bbs = f["part_bbs_matrix"][:]
            self.query_points = f["query_points_matrix"][:]
            self.query_labels = f["query_labels_matrix"][:]

        print(" Done.")

        # Create sample configurations
        self._create_sample_config()

    def _get_nan_model_indices(self):
        """
        Returns indices of models that contain NaN values in their part points.

        Returns:
            set: Set of model indices containing NaN values
        """
        # Find parts with NaN values
        nan_mask = np.isnan(self.part_points).any(axis=(1, 2))
        nan_part_indices = np.where(nan_mask)[0]

        if len(nan_part_indices) == 0:
            return set()

        # Find affected models by checking part ranges
        affected_models = set()
        for model_idx in range(len(self.model_ids)):
            start = self.part_slices[model_idx]
            end = self.part_slices[model_idx + 1]
            if np.any(nan_mask[start:end]):
                affected_models.add(model_idx)

        return affected_models

    def _create_sample_config(self):
        """
        Creating sample configurations while excluding models with NaN values.
        """
        nan_indices = self._get_nan_model_indices()
        if len(nan_indices) > 0 and self.rank == 0:
            print(f"Excluding {len(nan_indices)} models with NaN part pointclouds")

        self.sample_configs = []
        self.part_counts = []

        for model_idx, _ in enumerate(self.model_ids):
            if model_idx in nan_indices:
                continue

            # Get part information
            start_idx = self.part_slices[model_idx]
            end_idx = self.part_slices[model_idx + 1]
            n_parts = end_idx - start_idx

            # Calculate query configuration index
            query_config_idx = model_idx * (self.MAX_PART_DROP + 1)

            # Add original configuration (no dropped parts)
            self.sample_configs += [
                {
                    "model_idx": model_idx,
                    "query_config_idx": query_config_idx,
                    "part_slice": (start_idx, end_idx),
                    "dropped_part_idx": None,
                    "n_parts": n_parts,
                }
            ]
            self.part_counts += [n_parts]

            # Add part-drop configurations
            for drop_idx in range(self.MAX_PART_DROP):
                dropped_part_idx = self.part_drops[model_idx, drop_idx]
                if dropped_part_idx != -1:  # Valid part drop
                    self.sample_configs += [
                        {
                            "model_idx": model_idx,
                            "query_config_idx": query_config_idx + drop_idx + 1,
                            "part_slice": (start_idx, end_idx),
                            "dropped_part_idx": dropped_part_idx,
                            "n_parts": n_parts - 1,
                        }
                    ]
                    self.part_counts += [n_parts - 1]

    def _subsample_points(self, p, n_sub_points, labels=None):
        """
        Subsample points using random sampling with a fixed ratio.
        Ensures balanced labels (equal 1s and 0s) when labels are provided.

        Args:
            p: Points array of shape [N, 3]
            labels: Optional labels array of shape [N]. If provided, indicates query point processing.

        Returns:
            Subsampled points (and labels if provided)
        """
        p = torch.as_tensor(p)

        if labels is not None:
            # Convert labels to tensor if needed
            labels = torch.as_tensor(labels)

            # Get indices for each class
            idx_0 = torch.where(labels == 0)[0]
            idx_1 = torch.where(labels == 1)[0]

            n_points = len(p)

            assert (
                len(idx_0) == len(idx_1) == n_points // 2
            ), f"Invalid label distribution: {len(idx_0)} 0s, {len(idx_1)} 1s"

            # Sample equal numbers from each class
            n_per_class = n_sub_points // 2
            idx_0 = idx_0[torch.randperm(len(idx_0))[:n_per_class]]
            idx_1 = idx_1[torch.randperm(len(idx_1))[:n_per_class]]

            # Combine and shuffle indices
            idx = torch.cat([idx_0, idx_1])
            idx = idx[torch.randperm(len(idx))]

            p = p[idx]
            labels = labels[idx]

            assert len(p) == n_sub_points, f"Invalid subsampling length: {len(p)}"

            return p.numpy(), labels.numpy()
        else:
            # Part points - just random sampling
            idx = torch.randperm(len(p))[:n_sub_points]
            assert len(idx) == n_sub_points, f"Invalid subsampling length: {len(idx)}"

            return p[idx].numpy()

    def _subsample_queries(self, query_points, query_labels):
        """
        Subsample query points and labels to ensure fixed number of queries.
        """
        # Compute number of points to sample
        n_vol_points = self.num_queries // 2
        n_near_points = n_vol_points // (self.N_SAMPLING_FNS - 1)

        # First sample near-surface points
        near_points = []
        near_labels = []

        for i in range(self.N_SAMPLING_FNS - 1):
            sub_points, sub_labels = self._subsample_points(
                query_points[i], n_near_points, labels=query_labels[i]
            )
            near_points.append(sub_points)
            near_labels.append(sub_labels)

        near_points = np.concatenate(near_points)
        near_labels = np.concatenate(near_labels)

        # First sample volume points
        vol_points, vol_labels = self._subsample_points(
            query_points[-1], n_vol_points, labels=query_labels[-1]
        )

        # Combine near-surface and volume points
        query_points = np.vstack([vol_points, near_points])
        query_labels = np.hstack([vol_labels, near_labels])

        return query_points, query_labels

    def _apply_transformation(self, points, transformation_matrix):
        """
        Apply transformation matrix to points.

        Args:
            points: Points array of shape [N, 3]
            transformation_matrix: 4x4 transformation matrix

        Returns:
            Transformed points
        """
        # Convert to homogeneous coordinates
        if isinstance(points, torch.Tensor):
            homogeneous_points = torch.ones(
                (points.shape[0], 4), dtype=points.dtype, device=points.device
            )
            homogeneous_points[:, :3] = points

            # Convert matrix to tensor
            transformation_matrix = torch.from_numpy(transformation_matrix).to(
                dtype=points.dtype, device=points.device
            )

            # Apply transformation
            transformed_points = torch.matmul(
                homogeneous_points, transformation_matrix.T
            )

            # Return to 3D coordinates
            return transformed_points[:, :3]
        else:
            homogeneous_points = np.ones((points.shape[0], 4))
            homogeneous_points[:, :3] = points

            # Apply transformation
            transformed_points = np.matmul(homogeneous_points, transformation_matrix.T)

            # Return to 3D coordinates
            return transformed_points[:, :3]

    def _apply_transformation_to_bbs(self, bbs, transformation_matrix):
        """
        Apply transformation matrix to bounding boxes.

        Args:
            bbs: Bounding boxes array of shape [N_parts, 8, 3]
            transformation_matrix: 4x4 transformation matrix

        Returns:
            Transformed bounding boxes
        """
        # For each part
        transformed_bbs = []

        for part_bb in bbs:
            # Apply transformation to each corner
            transformed_bb = self._apply_transformation(part_bb, transformation_matrix)
            transformed_bbs.append(transformed_bb)

        if isinstance(bbs, torch.Tensor):
            return torch.stack(transformed_bbs)
        else:
            return np.stack(transformed_bbs)

    def __len__(self):
        return len(self.sample_configs)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Dictionary containing:
                - part_points: Part point clouds [N_parts, N_points, 3]
                - part_bbs: Part bounding boxes [N_parts, 8, 3]
                - query_points: Query points [5, N_queries, 3]
                - query_labels: Occupancy labels [N_queries]
                - model_id: Model identifier
        """
        config = self.sample_configs[idx]

        # Get part data
        start_idx, end_idx = config["part_slice"]
        part_points = self.part_points[start_idx:end_idx].copy()
        part_bbs = self.part_bbs[start_idx:end_idx].copy()

        n_points = part_points.shape[1]

        # Handle dropped part
        if config["dropped_part_idx"] is not None:
            mask = np.ones(end_idx - start_idx, dtype=bool)
            mask[config["dropped_part_idx"]] = False
            part_points = part_points[mask]
            part_bbs = part_bbs[mask]

            # Verify part removal
            assert (
                len(part_points) == config["n_parts"]
            ), "Mismatch in number of parts after dropping"

        # Get query data
        query_points = self.query_points[config["query_config_idx"]]
        query_labels = self.query_labels[config["query_config_idx"]]

        # Subsample query points
        query_points, query_labels = self._subsample_queries(query_points, query_labels)

        # Subsample surface part points if specified
        indices = np.random.choice(n_points, self.num_part_points, replace=False)
        part_points = part_points[:, indices]

        # Apply random transformation if enabled
        if self.random_transform:
            # Generate random transformation matrix
            min_scale = (
                1.0 / self.max_scale,
                1.0 / self.max_scale,
                1.0 / self.max_scale,
            )
            max_scale = (self.max_scale, self.max_scale, self.max_scale)
            transform_matrix = random_transformation_matrix(
                min_scale=min_scale,
                max_scale=max_scale,
                max_angle_x=self.rot_angle,
                max_angle_y=self.rot_angle,
                max_angle_z=self.rot_angle,
            )

            # Transform query points
            query_points = self._apply_transformation(query_points, transform_matrix)

            # Transform part bounding boxes
            part_bbs = self._apply_transformation_to_bbs(part_bbs, transform_matrix)

            # Note: part_points are NOT transformed as specified

        return {
            "part_points": torch.from_numpy(part_points).float(),
            "part_bbs": torch.from_numpy(part_bbs).float(),
            "query_points": torch.from_numpy(query_points).float(),
            "query_labels": torch.from_numpy(query_labels).float(),
            "model_id": self.model_ids[config["model_idx"]],
        }


class ShardedPartOccupancyDataset(PartOccupancyDataset):
    """
    Sharded version of PartOccupancyDataset that only loads a portion of the data
    based on the process rank.
    """

    def __init__(
        self,
        hdf5_path,
        rank,
        world_size,
        num_queries=2048,
        num_part_points=2048,
        random_transform=False,
        rot_angle=0.0,
        max_scale=0.0,
    ):

        # Initialize base attributes
        self.rank = rank
        self.num_queries = num_queries
        self.num_part_points = num_part_points

        # Store transformation parameters
        self.random_transform = random_transform
        self.rot_angle = np.radians(rot_angle)
        self.max_scale = max_scale

        # Load and validate HDF5 data
        if rank == 0:
            print("Loading HDF5 data to memory...")
            print()

        with h5py.File(hdf5_path, "r") as f:
            # Calculate model range for this shard
            total_models = len(f["model_ids"])
            models_per_shard = (total_models + world_size - 1) // world_size
            start_model = rank * models_per_shard
            end_model = min(start_model + models_per_shard, total_models)

            if start_model >= total_models:
                raise ValueError(f"Rank {rank} has no data to process")

            self.model_ids = f["model_ids"][start_model:end_model].astype("U")
            self.part_slices = f["part_slices"][start_model : end_model + 1]
            self.part_drops = f["part_drops"][start_model:end_model]

            # Load part data
            part_start = self.part_slices[0]
            part_end = self.part_slices[-1]
            self.part_points = f["part_points_matrix"][part_start:part_end]
            self.part_bbs = f["part_bbs_matrix"][part_start:part_end]

            # Make part slices local to this shard
            self.part_slices = self.part_slices - part_start

            # Load query data
            query_start = start_model * (self.MAX_PART_DROP + 1)
            query_end = end_model * (self.MAX_PART_DROP + 1)
            self.query_points = f["query_points_matrix"][query_start:query_end]
            self.query_labels = f["query_labels_matrix"][query_start:query_end]

        if rank == 0:
            print("Done.")

        # Create sample configurations
        self._create_sample_config()


def collate(batch):
    """
    Custom collate function for batching samples with fixed number of parts.

    Args:
        batch: List of sample dictionaries from the dataset

    Returns:
        Dictionary containing batched tensors:
            - part_points: [batch_size, n_parts, N_points, 3]
            - part_bbs: [batch_size, n_parts, 8, 3]
            - query_points: [batch_size, 5, N_queries, 3]
            - query_labels: [batch_size, 5, N_queries]
            - model_ids: List of str, length batch_size
    """
    if not batch:
        raise ValueError("Empty batch received")

    return {
        "part_points": torch.stack(
            [s["part_points"] for s in batch]
        ),  # [B, n_parts, N_points, 3]
        "part_bbs": torch.stack([s["part_bbs"] for s in batch]),  # [B, n_parts, 8, 3]
        "query_points": torch.stack(
            [s["query_points"] for s in batch]
        ),  # [B, 5, N_queries, 3]
        "query_labels": torch.stack(
            [s["query_labels"] for s in batch]
        ),  # [B, 5, N_queries]
        "model_ids": [s["model_id"] for s in batch],
    }
