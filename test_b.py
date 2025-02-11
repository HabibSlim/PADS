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
