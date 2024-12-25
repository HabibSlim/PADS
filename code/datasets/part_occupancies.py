import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class PartOccupancyDataset(Dataset):
    """
    Dataset for part-based occupancy prediction with part dropping.
    """

    MAX_PART_DROP = 16

    def __init__(
        self,
        hdf5_path,
        split="train",
        num_queries=2048,
        num_part_points=2048,
        num_replicas=1,
    ):
        """
        Initialize the dataset by loading HDF5 matrices into memory.

        Args:
            hdf5_path: Path to the HDF5 file containing the dataset
            split: Dataset split ('train', 'val', 'test')
            num_queries: Number of query points to sample (if None, uses all points)
            num_part_points: Number of points per part (if None, uses all points)
            num_replicas: Number of replicas for distributed training
        """
        self.num_queries = num_queries
        self.num_part_points = num_part_points
        self.num_replicas = num_replicas

        # Load and validate HDF5 data
        with h5py.File(hdf5_path, "r") as f:
            # Verify required datasets exist
            required_keys = [
                "model_ids",
                "part_slices",
                "part_drops",
                "part_points_matrix",
                "part_bbs_matrix",
                "query_points_matrix",
                "query_labels_matrix",
            ]
            missing_keys = [key for key in required_keys if key not in f]
            if missing_keys:
                raise ValueError(f"Missing required datasets: {missing_keys}")

            # Load data into memory
            self.model_ids = f["model_ids"][:].astype("U")
            self.part_slices = f["part_slices"][:]
            self.part_drops = f["part_drops"][:]
            self.part_points = f["part_points_matrix"][:]
            self.part_bbs = f["part_bbs_matrix"][:]
            self.query_points = f["query_points_matrix"][:]
            self.query_labels = f["query_labels_matrix"][:]

            # Validate dimensions
            n_models = len(self.model_ids)
            total_configs = self.query_points.shape[0]

            expected_configs = n_models * (self.MAX_PART_DROP + 1)

        # Create sample configurations
        self.sample_configs = []

        self.part_counts = []
        for model_idx, model_id in enumerate(self.model_ids):
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

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.sample_configs) * self.n_replicas

    def _subsample_queries(self, query_points, query_labels):
        """
        Subsample query points and labels to ensure fixed number of queries.
        """
        # Compute number of points to sample
        n_vol_points = self.num_queries // 2
        n_near_points = n_vol_points // 4

        query_points_all = []
        query_labels_all = []

        # First sample near-surface points
        for i in range(4):
            indices = np.random.choice(n_near_points, n_near_points, replace=False)
            query_points_all += [query_points[i, indices]]
            query_labels_all += [query_labels[i, indices]]

        query_points_sub = np.stack(query_points_all).reshape(-1, 3)
        query_labels_sub = np.stack(query_labels_all).reshape(-1)

        # Then sample volume points
        indices = np.random.choice(n_vol_points, n_vol_points, replace=False)
        query_points = np.vstack([query_points_sub, query_points[4, indices]])
        query_labels = np.hstack([query_labels_sub, query_labels[4, indices]])

        return query_points, query_labels

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

        # Subsample part points if specified
        indices = np.random.choice(n_points, self.num_part_points, replace=False)
        part_points = part_points[:, indices]

        return {
            "part_points": torch.from_numpy(part_points).float(),
            "part_bbs": torch.from_numpy(part_bbs).float(),
            "query_points": torch.from_numpy(query_points).float(),
            "query_labels": torch.from_numpy(query_labels).float(),
            "model_id": self.model_ids[config["model_idx"]],
        }


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
