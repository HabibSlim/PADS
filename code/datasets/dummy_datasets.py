import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader


class DummyPartDataset(Dataset):
    def __init__(
        self,
        num_samples=1000,
        num_part_points=1024,
        num_queries=2048,
        num_replicas=1,
        max_parts=5,
        full_parts=False
    ):
        self.num_samples = num_samples
        self.max_parts = max_parts
        self.num_part_points = num_part_points
        self.num_queries = num_queries
        self.num_replicas = num_replicas

        # Generate random number of parts for each sample
        if full_parts:
            self.part_counts = torch.full((self.num_samples,), self.max_parts)
        else:
            self.part_counts = torch.randint(
                low=1, high=self.max_parts + 1, size=(self.num_samples,)
            )

        # Generate random model ids
        self.model_ids = torch.randint(low=0, high=10, size=(self.num_samples,))

    def __len__(self):
        return self.num_samples * self.num_replicas

    def __getitem__(self, idx):
        # Get number of parts for this batch
        num_parts = self.part_counts[idx].item()

        # Random pointcloud for each part (P x N x 3)
        part_points = torch.randn(num_parts, self.num_part_points, 3)

        # Random bounding boxes (P x 8 x 3)
        centers = torch.randn(num_parts, 1, 3)
        dims = torch.rand(num_parts, 1, 3) + 0.5  # Add 0.5 to avoid too small boxes

        # Create corner offsets for unit cube
        corners = (
            torch.tensor(
                [
                    [-1, -1, -1],
                    [+1, -1, -1],
                    [-1, +1, -1],
                    [+1, +1, -1],
                    [-1, -1, +1],
                    [+1, -1, +1],
                    [-1, +1, +1],
                    [+1, +1, +1],
                ]
            )
            * 0.5
        )

        # Scale and translate corners for each part
        part_bbs = (corners.view(1, 8, 3) * dims + centers).float()

        # Random occupancy points and labels
        query_points = torch.randn(self.num_queries, 3)
        query_labels = torch.randint(0, 2, (self.num_queries,))

        return {
            "num_parts": num_parts,
            "part_points": part_points,
            "part_bbs": part_bbs,
            "query_points": query_points,
            "query_labels": query_labels,
            "model_id": self.model_ids[idx].item(),
        }


def collate_dummy(batch):
    """
    Collate function for batches with same number of parts.
    """
    batch_size = len(batch)

    # Stack tensors directly - no padding needed since same number of parts in batch
    part_points = torch.stack([s["part_points"] for s in batch])
    part_bbs = torch.stack([s["part_bbs"] for s in batch])
    query_points = torch.stack([s["query_points"] for s in batch])
    query_labels = torch.stack([s["query_labels"] for s in batch])
    model_ids = torch.tensor([s["model_id"] for s in batch])

    return {
        "part_points": part_points,
        "part_bbs": part_bbs,
        "query_points": query_points,
        "query_labels": query_labels,
        "model_ids": model_ids,
    }
