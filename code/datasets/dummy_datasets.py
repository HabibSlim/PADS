"""
Dummy dataset for part pointclouds with varying number of parts.
"""

import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader


class DummyPartDataset(Dataset):
    def __init__(
        self, num_samples=1000, max_parts=5, num_points=1024, num_occ=2048, n_replicas=1
    ):
        self.num_samples = num_samples
        self.max_parts = max_parts
        self.num_points = num_points
        self.num_occ = num_occ
        self.n_replicas = n_replicas

    def __len__(self):
        return self.num_samples * self.n_replicas

    def __getitem__(self, idx):
        # Random number of parts (1 to max_parts)
        num_parts = torch.randint(1, self.max_parts + 1, (1,)).item()

        # Random pointcloud for each part (P x N x 3)
        part_points = torch.randn(num_parts, self.num_points, 3)

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
        bounding_boxes = (corners.view(1, 8, 3) * dims + centers).float()

        # Random occupancy points and labels
        occ_points = torch.randn(self.num_occ, 3)
        occ_labels = torch.randint(0, 2, (self.num_occ,))

        return {
            "num_parts": num_parts,
            "part_points": part_points,
            "bounding_boxes": bounding_boxes,
            "occupancy_points": occ_points,
            "occupancy_labels": occ_labels,
        }


def collate_varying_parts(batch):
    """
    Collate function to handle varying number of parts by padding.
    """
    batch_size = len(batch)
    max_parts = max(sample["num_parts"] for sample in batch)
    N = batch[0]["part_points"].shape[1]
    N_occ = batch[0]["occupancy_points"].shape[1]

    # Initialize padded tensors
    padded_points = torch.zeros(batch_size, max_parts, N, 3)
    padded_boxes = torch.zeros(batch_size, max_parts, 8, 3)
    num_parts = torch.zeros(batch_size, dtype=torch.long)
    occ_points = torch.stack([s["occupancy_points"] for s in batch])
    occ_labels = torch.stack([s["occupancy_labels"] for s in batch])

    # Fill padded tensors
    for i, sample in enumerate(batch):
        P = sample["num_parts"]
        num_parts[i] = P
        padded_points[i, :P] = sample["part_points"]
        padded_boxes[i, :P] = sample["bounding_boxes"]

    return {
        "num_parts": num_parts,
        "part_points": padded_points,
        "bounding_boxes": padded_boxes,
        "occupancy_points": occ_points,
        "occupancy_labels": occ_labels,
    }
