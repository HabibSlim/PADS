"""
Occupancies dataset.
"""

import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from datasets.sampling import normalize_pc
from torch.utils.data import Dataset


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
        get_part_points=False,
        normalize_part_points=False,
        n_replicas=1,
    ):
        exclude_types = set(exclude_types) if exclude_types else set()
        self.shuffle_parts = shuffle_parts
        self.get_part_points = get_part_points
        self.normalize_part_points = normalize_part_points
        self.n_replicas = n_replicas

        # Load file list
        file_list = "capped_list.json" if cap_parts else "full_list.json"
        file_list = json.load(open(os.path.join(data_dir, file_list)))
        latents_dir = os.path.join(data_dir, "latents")
        bbs_dir = os.path.join(data_dir, "bounding_boxes")
        points_dir = os.path.join(data_dir, "part_points")

        # Load the split
        if split is not None:
            self.split_name = split
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
                part_points_f = f[:6]
                if self.get_part_points:
                    final_list += [
                        [
                            k + ".npy"
                            for k in [f, bb_coords_f, bb_labels_f, part_points_f]
                        ]
                    ]
                else:
                    final_list += [[k + ".npy" for k in [f, bb_coords_f, bb_labels_f]]]

        # Create a list of file paths
        file_list = final_list
        file_list.sort()
        self.file_list = file_list
        self.file_tuples = []
        for idx in range(len(file_list)):
            # Unpack file paths
            file_paths = (
                [os.path.join(latents_dir, file_list[idx][0])]
                + [os.path.join(bbs_dir, f) for f in file_list[idx][1:3]]
                + [None]
            )

            if self.get_part_points:
                file_paths[3] = os.path.join(
                    points_dir, file_list[idx][3]
                )  # part points

            # Extract model ID from the filename
            basename = os.path.basename(file_paths[0])
            cls_code = basename.split("_")[0:2][0]
            model_id = cls_code + basename.split("_")[0:2][1]
            model_id = int(model_id, 16)

            # Convert hex to int
            cls_label = int(cls_code, 16)

            self.file_tuples += [(*file_paths, cls_label, model_id)]

        if filter_n_ids is not None:
            # Only keep the samples corresponding to N=filter_n_ids distinct model IDs
            unique_model_ids = list(set(t[-1] for t in self.file_tuples))
            random.shuffle(unique_model_ids)
            selected_model_ids = set(unique_model_ids[:filter_n_ids])

            self.file_tuples = [
                t for t in self.file_tuples if t[-1] in selected_model_ids
            ]
            filtered_file_paths = [t[0] for t in self.file_tuples]
            self.file_list = [
                sublist
                for sublist in self.file_list
                if os.path.join(latents_dir, sublist[0]) in filtered_file_paths
            ]

        self.rng = torch.Generator()
        self.rng_counter = 0

    def __len__(self):
        if self.split_name != "train":
            return len(self.file_tuples)
        else:
            return len(self.file_tuples) * self.n_replicas

    def __getitem__(self, idx):
        # Unpack file paths
        latent_f, bb_coords_f, bb_labels_f, part_points_f, cls_label, model_id = (
            self.file_tuples[idx]
        )

        # Loading latent and bounding box data
        latent = np.load(latent_f)
        bb_coords = np.load(bb_coords_f)
        bb_labels = np.load(bb_labels_f)

        # Convert numpy array to torch tensor
        latent_tensor = torch.from_numpy(latent).float()
        bb_coords_tensor = torch.from_numpy(bb_coords).float()
        bb_labels_tensor = torch.from_numpy(bb_labels).long()
        cls_label_tensor = torch.tensor(cls_label).long()

        if self.get_part_points:
            part_points = np.load(part_points_f, allow_pickle=True)
            part_points_tensor = torch.from_numpy(part_points).float()
            # Normalize each part point cloud
            if self.normalize_part_points:
                for k in range(part_points_tensor.shape[0]):
                    part_points_tensor[k] = normalize_pc(
                        part_points_tensor[k].unsqueeze(0),
                        method="per_axis",
                    ).squeeze()

        # Shuffle the order of parts if self.shuffle is True
        if self.shuffle_parts:
            self.rng.manual_seed(model_id + self.rng_counter)

            num_parts = bb_coords_tensor.size(0)
            shuffle_indices = torch.randperm(num_parts, generator=self.rng)
            bb_coords_tensor = bb_coords_tensor[shuffle_indices]
            bb_labels_tensor = bb_labels_tensor[shuffle_indices]
            if self.get_part_points:
                part_points_tensor = part_points_tensor[shuffle_indices]

        # Pad bb coords and labels
        pad_size = self.PART_CAP - bb_coords_tensor.size(0)

        # Pad the tensors
        bb_coords_tensor = F.pad(bb_coords_tensor, (0, 0, 0, 0, 0, pad_size))
        bb_labels_tensor = F.pad(bb_labels_tensor, (0, pad_size), value=-1)
        if self.get_part_points:
            part_points_tensor = F.pad(part_points_tensor, (0, 0, 0, 0, 0, pad_size))

        # Extract metadata from filename
        meta = os.path.basename(latent_f).split(".")[0]

        if self.get_part_points:
            return (
                latent_tensor,
                bb_coords_tensor,
                bb_labels_tensor,
                part_points_tensor,
                cls_label_tensor,
                meta,
            )
        else:
            return (
                latent_tensor,
                bb_coords_tensor,
                bb_labels_tensor,
                cls_label_tensor,
                meta,
            )
