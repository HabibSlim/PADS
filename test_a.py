import h5py
import numpy as np


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
        truncate_unit_cube=True,
    ):
        # Initialize base attributes
        self.num_queries = num_queries
        self.num_part_points = num_part_points
        self.truncate_unit_cube = truncate_unit_cube
        self.rank = rank

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
        self.create_sample_config()

    def get_nan_model_indices(self):
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

    def create_sample_config(self):
        """
        Creating sample configurations while excluding models with NaN values.
        """
        nan_indices = self.get_nan_model_indices()
        if len(nan_indices) > 0 and self.rank == 0:
            print(f"Excluding {len(nan_indices)} models with NaN values")

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
