"""
Graph edits dataset loader.
"""

import h5py
import json
import os
import torch
import numpy as np
from torch.utils import data
from collections import defaultdict

GRAPH__N_POINTS = 8192


class GraphEdits(data.Dataset):
    """
    GraphEdits dataset for loading NPY pointclouds.
    """

    def __init__(
        self,
        dataset_folder,
        dataset_type,
        split,
        transform=None,
        sampling=True,
        num_samples=4096,
        pc_size=2048,
        replica=1,
        max_edge_level=None,
        get_voxels=False,
        fetch_keys=False,
        fetch_intensity=False,
    ):
        self.get_voxels = get_voxels
        self.pc_size = pc_size

        self.transform = transform
        self.num_samples = num_samples
        self.sampling = sampling or GRAPH__N_POINTS != self.pc_size
        self.split = split

        if self.get_voxels:
            self.data_folder = os.path.join(dataset_folder, "voxels")
        else:
            self.data_folder = os.path.join(dataset_folder, "pc")
        self.replica = replica
        self.fetch_keys = fetch_keys

        dataset_entry = json.load(
            open(
                os.path.join(
                    dataset_folder, dataset_type, dataset_type + "_" + split + ".json"
                )
            )
        )
        self.dataset = []
        for node_key, entry in dataset_entry.items():
            cur_edge_level = int(entry["edge_level"])
            if max_edge_level is not None and cur_edge_level > max_edge_level:
                continue
            node_a, node_b = node_key.split("_")
            self.dataset += [
                {
                    "edit_key": node_key,
                    "node_a": node_a,
                    "node_b": node_b,
                    "prompt": entry["prompt"],
                }
            ]

    def load_npy(self, path, scale):
        data = np.load(path)
        surface = data.astype(np.float32)
        return surface * scale

    def load_voxels(self, path):
        data = np.load(path)
        voxels = np.unpackbits(data, axis=-1)
        voxels = np.expand_dims(voxels, axis=0)
        return voxels

    def resample_points(self, points):
        ind = np.random.default_rng().choice(
            points.shape[0], self.pc_size, replace=False
        )
        return points[ind]

    def __getitem__(self, idx):
        idx = idx % len(self.dataset)
        node_a = self.dataset[idx]["node_a"]
        node_b = self.dataset[idx]["node_b"]
        prompt = self.dataset[idx]["prompt"]

        node_a = os.path.join(self.data_folder, node_a + ".npy")
        node_b = os.path.join(self.data_folder, node_b + ".npy")

        # Load pointclouds a/b
        if self.get_voxels:
            node_a = self.load_voxels(node_a)
            node_b = self.load_voxels(node_b)
        else:
            scale = 1.0
            node_a = self.load_npy(node_a, scale)
            node_b = self.load_npy(node_b, scale)

            if self.sampling:
                node_a = self.resample_points(node_a)
                node_b = self.resample_points(node_b)

        if self.transform:
            node_a = self.transform(node_a)
            node_b = self.transform(node_b)

        # Convert to torch tensors
        node_a = torch.from_numpy(node_a).type(torch.float32)
        node_b = torch.from_numpy(node_b).type(torch.float32)

        if not self.fetch_keys:
            return node_a, node_b, prompt
        else:
            return self.dataset[idx]["edit_key"], node_a, node_b, prompt

    def __len__(self):
        if self.split != "train":
            return len(self.dataset)
        else:
            return len(self.dataset) * self.replica


def decode_json_dset(dset):
    dset = dset[:][0].decode("utf-8")
    return json.loads(dset)


class GraphEditsEmbeds(data.Dataset):
    """
    GraphEdits dataset for loading pre-extracted shape/text embeddings.
    """

    def __init__(
        self,
        dataset_folder,
        dataset_type,
        split,
        transform=None,
        alt_ae_embeds=None,
        replica=1,
        fetch_keys=False,
        fetch_intensity=False,
        fetch_text_prompts=False,
        fetch_edge_dict=False,
    ):
        self.transform = transform
        self.split = split

        self.replica = replica
        self.fetch_keys = fetch_keys
        self.fetch_intensity = fetch_intensity
        self.fetch_text_prompts = fetch_text_prompts
        self.fetch_edge_dict = fetch_edge_dict

        if alt_ae_embeds is not None:
            ae_str = "__%s" % alt_ae_embeds
        else:
            ae_str = ""

        hdf5_file = os.path.join(
            dataset_folder, dataset_type, "embeddings_%s%s.hdf5" % (split, ae_str)
        )
        hdf5_f = h5py.File(hdf5_file, "r")

        # Load everything in RAM
        self.shape_embeds = (
            torch.tensor(hdf5_f["shape_embeds"][:]).to("cpu").type(torch.float32)
        )
        if not self.fetch_text_prompts:
            self.text_embeds = (
                torch.tensor(hdf5_f["text_embeds"][:]).to("cpu").type(torch.float32)
            )
        self.key_to_shape_embeds = decode_json_dset(hdf5_f["key_to_shape_embeds"])
        self.key_pair_to_text_embeds = decode_json_dset(
            hdf5_f["key_pair_to_text_embeds"]
        )

        # Load JSON dataset
        dataset_entry = json.load(
            open(
                os.path.join(
                    dataset_folder,
                    dataset_type,
                    dataset_type + "_" + split + ".json",
                )
            )
        )

        self.dataset = []
        for node_key in self.key_pair_to_text_embeds:
            node_a, node_b = node_key.split("_")
            self.dataset += [
                {
                    "edit_key": node_key,
                    "node_a": node_a,
                    "node_b": node_b,
                }
            ]
            if self.fetch_edge_dict:
                self.dataset[-1]["edge_dict"] = dataset_entry[node_key]["edge_dict"]
            if self.fetch_intensity:
                self.dataset[-1]["edge_intensity"] = dataset_entry[node_key][
                    "edge_intensity"
                ]
            if self.fetch_text_prompts:
                self.dataset[-1]["prompt"] = dataset_entry[node_key]["prompt"]

        hdf5_f.close()

    def __getitem__(self, idx):
        idx = idx % len(self.dataset)
        entry = self.dataset[idx]
        edit_key = entry["edit_key"]
        node_a = entry["node_a"]
        node_b = entry["node_b"]

        node_a_idx = self.key_to_shape_embeds[node_a]
        node_b_idx = self.key_to_shape_embeds[node_b]

        node_a = self.shape_embeds[node_a_idx]
        node_b = self.shape_embeds[node_b_idx]

        if not self.fetch_text_prompts:
            text_embed_idx = self.key_pair_to_text_embeds[edit_key]
            text_embed = self.text_embeds[text_embed_idx]

        # Make a list of elements to return and handle all cases
        return_list = []
        if self.fetch_keys:
            return_list.append(edit_key)
        return_list += [node_a, node_b]

        # Append text prompt/embedding
        if self.fetch_text_prompts:
            return_list.append(entry["prompt"])
        else:
            return_list.append(text_embed)

        if self.fetch_intensity:
            return_list.append(entry["edge_intensity"])

        if self.fetch_edge_dict:
            return_list.append(entry["edge_dict"])

        return tuple(return_list)

    def __len__(self):
        if self.split != "train":
            return len(self.dataset)
        else:
            return len(self.dataset) * self.replica


class GraphEditsEmbedsNRL(data.Dataset):
    """
    GraphEdits dataset for loading pre-extracted shape/text embeddings.
    Neural Listener version of the dataset.
    """

    def __init__(
        self,
        dataset_folder,
        dataset_type,
        split,
        transform=None,
        alt_ae_embeds=None,
        replica=1,
        fetch_keys=False,
        fetch_intensity=False,
        fetch_text_prompts=False,
    ):
        self.transform = transform
        self.split = split

        self.replica = replica
        self.fetch_keys = fetch_keys
        self.fetch_text_prompts = fetch_text_prompts
        self.fetch_intensity = True
        self.fetch_edge_dict = True

        if alt_ae_embeds is not None:
            ae_str = "__%s" % alt_ae_embeds
        else:
            ae_str = ""

        hdf5_file = os.path.join(
            dataset_folder, dataset_type, "embeddings_%s%s.hdf5" % (split, ae_str)
        )
        hdf5_f = h5py.File(hdf5_file, "r")

        # Load everything in RAM
        self.shape_embeds = (
            torch.tensor(hdf5_f["shape_embeds"][:]).to("cpu").type(torch.float32)
        )
        if not self.fetch_text_prompts:
            self.text_embeds = (
                torch.tensor(hdf5_f["text_embeds"][:]).to("cpu").type(torch.float32)
            )
        self.key_to_shape_embeds = decode_json_dset(hdf5_f["key_to_shape_embeds"])
        self.key_pair_to_text_embeds = decode_json_dset(
            hdf5_f["key_pair_to_text_embeds"]
        )

        # Load JSON dataset
        # if self.fetch_intensity or self.fetch_text_prompts:
        dataset_entry = json.load(
            open(
                os.path.join(
                    dataset_folder,
                    dataset_type,
                    dataset_type + "_" + split + ".json",
                )
            )
        )

        self.dataset = []
        for node_key in self.key_pair_to_text_embeds:
            node_a, node_b = node_key.split("_")
            self.dataset += [
                {
                    "edit_key": node_key,
                    "node_a": node_a,
                    "node_b": node_b,
                }
            ]
            if self.fetch_edge_dict:
                self.dataset[-1]["edge_dict"] = dataset_entry[node_key]["edge_dict"]
            if self.fetch_text_prompts:
                self.dataset[-1]["prompt"] = dataset_entry[node_key]["prompt"]
            if self.fetch_intensity:
                self.dataset[-1]["edge_intensity"] = dataset_entry[node_key][
                    "edge_intensity"
                ]

        hdf5_f.close()

    def __getitem__(self, idx):
        idx = idx % len(self.dataset)
        entry = self.dataset[idx]
        edit_key = entry["edit_key"]
        node_a = entry["node_a"]
        node_b = entry["node_b"]

        node_a_idx = self.key_to_shape_embeds[node_a]
        node_b_idx = self.key_to_shape_embeds[node_b]

        node_a = self.shape_embeds[node_a_idx]
        node_b = self.shape_embeds[node_b_idx]

        if not self.fetch_text_prompts:
            text_embed_idx = self.key_pair_to_text_embeds[edit_key]
            text_embed = self.text_embeds[text_embed_idx]

        # Flip x_a and x_b with 50% probability
        if torch.rand(1) > 0.5:
            node_a, node_b = node_b, node_a
            label = torch.tensor([0.0])
        else:
            label = torch.tensor([1.0])

        # Make a list of elements to return and handle all cases
        return_list = []
        if self.fetch_keys:
            return_list.append(edit_key)
        return_list += [node_a, node_b]

        # Append text prompt/embedding
        if self.fetch_text_prompts:
            return_list.append(entry["prompt"])
        else:
            return_list.append(text_embed)

        if self.fetch_intensity:
            return_list.append(entry["edge_intensity"])

        if self.fetch_edge_dict:
            return_list.append(entry["edge_dict"])

        return_list += [label]

        return tuple(return_list)

    def __len__(self):
        if self.split != "train":
            return len(self.dataset)
        else:
            return len(self.dataset) * self.replica


class GraphEditsEmbedsChained(data.Dataset):
    """
    GraphEdits dataset for loading pre-extracted shape/text embeddings.
    Chained version of the dataloader.
    """

    def __init__(
        self,
        dataset_folder,
        dataset_type,
        split,
        chain_length,
        transform=None,
        alt_ae_embeds=None,
        replica=1,
        fetch_keys=False,
        fetch_intensity=False,
        fetch_text_prompts=False,
        fetch_edge_dict=False,
    ):
        self.transform = transform
        self.split = split

        self.replica = replica
        self.fetch_keys = fetch_keys
        self.fetch_intensity = fetch_intensity
        self.fetch_text_prompts = fetch_text_prompts
        self.fetch_edge_dict = fetch_edge_dict

        if alt_ae_embeds is not None:
            ae_str = "__%s" % alt_ae_embeds
        else:
            ae_str = ""

        hdf5_file = os.path.join(
            dataset_folder, dataset_type, "embeddings_%s%s.hdf5" % (split, ae_str)
        )
        hdf5_f = h5py.File(hdf5_file, "r")

        # Load everything in RAM
        self.shape_embeds = (
            torch.tensor(hdf5_f["shape_embeds"][:]).to("cpu").type(torch.float32)
        )
        if not self.fetch_text_prompts:
            self.text_embeds = (
                torch.tensor(hdf5_f["text_embeds"][:]).to("cpu").type(torch.float32)
            )
        self.key_to_shape_embeds = decode_json_dset(hdf5_f["key_to_shape_embeds"])
        self.key_pair_to_text_embeds = decode_json_dset(
            hdf5_f["key_pair_to_text_embeds"]
        )

        # Load JSON dataset
        dataset_entry = json.load(
            open(
                os.path.join(
                    dataset_folder,
                    dataset_type,
                    dataset_type + "_" + split + ".json",
                )
            )
        )

        get_chain_id = lambda x: ("_".join(x.split("_")[:-1]), x.split("_")[-1])

        self.dataset = []
        chain_id_counter = defaultdict(int)
        for node_key in self.key_pair_to_text_embeds:
            if dataset_entry[node_key]["chain_length"] != int(chain_length):
                continue
            chain_id = dataset_entry[node_key]["chain_id"]
            chain_root_id, chain_id = get_chain_id(chain_id)
            chain_id_counter[chain_root_id] += 1

            new_chain_id = chain_root_id + "_%02d" % int(chain_id)

            # node_id = int(node_key.split("_")[-1])
            node_a, node_b = node_key.split("_")
            self.dataset += [
                {
                    "edit_key": node_key,
                    "node_a": node_a,
                    "node_b": node_b,
                    "chain_id": new_chain_id,
                }
            ]
            if self.fetch_edge_dict:
                self.dataset[-1]["edge_dict"] = dataset_entry[node_key]["edge_dict"]
            if self.fetch_intensity:
                self.dataset[-1]["edge_intensity"] = dataset_entry[node_key][
                    "edge_intensity"
                ]
            if self.fetch_text_prompts:
                self.dataset[-1]["prompt"] = dataset_entry[node_key]["prompt"]

        # Filter the correct chain length
        self.dataset = [
            entry
            for entry in self.dataset
            if chain_id_counter[get_chain_id(entry["chain_id"])[0]] == int(chain_length)
        ]

        # Sort dataset by chain_id
        self.dataset = sorted(self.dataset, key=lambda x: x["chain_id"])

        hdf5_f.close()

    def __getitem__(self, idx):
        idx = idx % len(self.dataset)
        entry = self.dataset[idx]
        edit_key = entry["edit_key"]
        chain_id = entry["chain_id"]
        node_a = entry["node_a"]
        node_b = entry["node_b"]

        node_a_idx = self.key_to_shape_embeds[node_a]
        node_b_idx = self.key_to_shape_embeds[node_b]

        node_a = self.shape_embeds[node_a_idx]
        node_b = self.shape_embeds[node_b_idx]

        if not self.fetch_text_prompts:
            text_embed_idx = self.key_pair_to_text_embeds[edit_key]
            text_embed = self.text_embeds[text_embed_idx]

        # Make a list of elements to return and handle all cases
        return_list = [chain_id]
        if self.fetch_keys:
            return_list.append(edit_key)
        return_list += [node_a, node_b]

        if self.fetch_edge_dict:
            return_list.append(entry["edge_dict"])

        # Append text prompt/embedding
        if self.fetch_text_prompts:
            return_list.append(entry["prompt"])
        else:
            return_list.append(text_embed)

        if self.fetch_intensity:
            return_list.append(entry["edge_intensity"])

        return tuple(return_list)

    def __len__(self):
        if self.split != "train":
            return len(self.dataset)
        else:
            return len(self.dataset) * self.replica
