"""
Extracting features into HDF5 files for each split.
"""
import argparse
import datetime
import h5py
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from transformers import AutoTokenizer, CLIPTextModel, BertTokenizer, BertModel

import util.misc as misc
import models_ae
from engine_node2node import get_text_embeddings
from util.datasets import build_shape_surface_occupancy_dataset



def get_args_parser():
    parser = argparse.ArgumentParser("Latent Diffusion", add_help=False)

    # Model parameters
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument(
        "--text_model_name",
        type=str,
        help="Text model name to use",
    )
    parser.add_argument(
        "--ae",
        type=str,
        metavar="MODEL",
        help="Name of autoencoder",
    )
    parser.add_argument(
        "--ae-latent-dim",
        type=int,
        default=512*8,
        help="AE latent dimension",
    )
    parser.add_argument(
        "--ae-pth",
        required=True,
        help="Autoencoder checkpoint"
    )
    parser.add_argument(
        "--point_cloud_size",
        default=2048,
        type=int,
        help="input size"
    )
    parser.add_argument(
        "--fetch_keys",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--use_embeds",
        action="store_true",
        default=False,
    )

    # Dataset parameters
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["graphedits"],
        help="dataset name",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="dataset path",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        help="dataset type",
    )
    parser.add_argument(
        "--max_edge_level",
        default=None,
        type=int,
        help="maximum edge level to use",
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_workers", default=60, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )

    return parser


def get_split_embeddings(args, data_loader):
    """
    Extract embeddings for a given split.
    """
    text_latent_dim = 512 if args.use_clip else 768

    # Create stacked numpy arrays to store embeddings
    B = args.batch_size
    n_batches = len(data_loader)
    n_entries = n_batches * B
    embeds_xa   = np.zeros((n_entries, args.ae_latent_dim))
    embeds_xb   = np.zeros((n_entries, args.ae_latent_dim))
    embeds_text = np.zeros((n_entries, text_latent_dim))
    all_keys = []

    # Iterate over the dataset and extract text embeddings
    for k, (edit_keys, nodes_a, nodes_b, prompts_ab) in enumerate(data_loader):
        nodes_a = nodes_a.to(device, non_blocking=True)
        nodes_b = nodes_b.to(device, non_blocking=True)
        embeds_ab = get_text_embeddings(text_model=text_model,
                                        tokenizer=tokenizer,
                                        texts=prompts_ab,
                                        device=device)

        with torch.cuda.amp.autocast(enabled=False):
            with torch.no_grad():
                _, x_a = ae.encode(nodes_a)
                _, x_b = ae.encode(nodes_b)

        # Move batch to CPU and convert to numpy
        embeds_ab = embeds_ab.cpu().numpy()
        x_a = x_a.cpu().numpy()
        x_b = x_b.cpu().numpy()

        # Store to stacked arrays
        embeds_xa[k*B:(k+1)*B] = x_a.reshape(B, -1)
        embeds_xb[k*B:(k+1)*B] = x_b.reshape(B, -1)
        embeds_text[k*B:(k+1)*B] = embeds_ab.reshape(B, -1)
        all_keys += edit_keys
    
    return edit_keys, embeds_xa, embeds_xb, embeds_text


def extract_embeddings(split, data_loader):
    """
    Extract embeddings and remap to shape/edit keys.
    """
    # Extract embeddings
    edit_keys, embeds_xa, embeds_xb, text_embeds = get_split_embeddings(args, data_loader)

    edit_keys_sp = [k.split('_') for k in edit_keys]
    keys_node_a = [k[0] for k in edit_keys_sp]
    keys_node_b = [k[1] for k in edit_keys_sp]

    text_latent_dim = 512 if args.use_clip else 768
    
    # Map node keys to indices
    node_a_to_idx = {k: i for i, k in enumerate(keys_node_a)}
    node_b_to_idx = {k: i for i, k in enumerate(keys_node_b)}
    all_nodes = list(set(keys_node_a + keys_node_b))

    # Build a matrix with all the embeddings
    # using the indices
    shape_embeds = np.zeros((len(all_nodes), args.ae_latent_dim))
    k = 0
    key_to_shape_embeds = {}
    for node_key in node_a_to_idx:
        idx = node_a_to_idx[node_key]
        shape_embeds[k] = embeds_xa[idx]
        key_to_shape_embeds[node_key] = k
        k += 1

    # Remove all nodes already added from node_b
    for node_key in (node_b_to_idx.keys() - node_a_to_idx.keys()):
        idx = node_b_to_idx[node_key]
        shape_embeds[k] = embeds_xb[idx]
        key_to_shape_embeds[node_key] = k
        k += 1

    # Double check that everything is correct
    # Iterate on edit_keys
    print("Checking shape embeddings...")
    for node_a, node_b in edit_keys_sp:
        assert node_a in key_to_shape_embeds
        assert node_b in key_to_shape_embeds

        # Check that embeddings are correct
        idx_a = key_to_shape_embeds[node_a]
        idx_b = key_to_shape_embeds[node_b]

        assert np.allclose(shape_embeds[idx_a], embeds_xa[node_a_to_idx[node_a]])
        assert np.allclose(shape_embeds[idx_b], embeds_xb[node_b_to_idx[node_b]])
    print("Done!")

    key_pair_to_text_embeds = {key_pair : k for k, key_pair in enumerate(edit_keys)}

    # Double check that everything is correct
    # Iterate on edit_keys
    print("Checking text embeddings...")
    for key_pair in edit_keys:
        assert key_pair in key_pair_to_text_embeds

    print("Done!")

    return shape_embeds, key_to_shape_embeds, text_embeds, key_pair_to_text_embeds


def create_hdf5(args, split, shape_embeds, key_to_shape_embeds, embeds_text, key_pair_to_text_embeds):
    """
    Create HDF5 file with the embeddings
    """
    # Create HDF5 file
    hdf5_path = os.path.join(args.data_path, args.data_type, "embeddings_%s.hdf5" % split)
    # If exists: delete
    if os.path.exists(hdf5_path):
        os.remove(hdf5_path)
        print("Deleted existing HDF5 file %s" % hdf5_path)
    print("Creating HDF5 file %s" % hdf5_path)
    f = h5py.File(hdf5_path, "w")

    # Create datasets
    f.create_dataset("shape_embeds", data=shape_embeds)
    f.create_dataset("text_embeds", data=embeds_text)
    f.create_dataset("key_to_shape_embeds",
                     data=json.dumps(key_to_shape_embeds),
                     shape=(1,),
                     dtype=h5py.string_dtype(encoding="utf-8"))
    f.create_dataset("key_pair_to_text_embeds",
                     data=json.dumps(key_pair_to_text_embeds),
                     shape=(1,),
                     dtype=h5py.string_dtype(encoding="utf-8"))
    f.close()

    print("Done!")


def main(args):
    """
    Main feature extraction + HDF5 packing routine.
    """
    # --------------------
    args.use_clip = "clip" in args.text_model_name
    device = torch.device(args.device)

    # Fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    args.fetch_keys = True
    dataset_train = build_shape_surface_occupancy_dataset("train", args=args)
    dataset_val = build_shape_surface_occupancy_dataset("val", args=args)

    # Create data loaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    # --------------------

    # Instantiate autoencoder
    ae = models_ae.__dict__[args.ae]()
    ae.eval()
    print("Loading autoencoder %s" % args.ae_pth)
    ae.load_state_dict(torch.load(args.ae_pth, map_location="cpu")["model"])
    ae.to(device)

    # Initialize text CLIP model
    if args.use_clip:
        # Instantiate tokenizer + CLIP model
        tokenizer = AutoTokenizer.from_pretrained(args.text_model_name)
        text_model = CLIPTextModel.from_pretrained(args.text_model_name).to(device)
    else:
        # Instantiate BERT model and create linear projection layer
        tokenizer = BertTokenizer.from_pretrained(args.text_model_name)
        text_model = BertModel.from_pretrained(args.text_model_name).to(device)

    # Extracting training embeddings
    print("Extracting training embeddings...")
    shape_embeds_train, key_to_shape_embeds_train, text_embeds_train, key_pair_to_text_embeds_train = extract_embeddings("train", data_loader_train)
    create_hdf5(args, "train", shape_embeds_train, key_to_shape_embeds_train, text_embeds_train, key_pair_to_text_embeds_train)
    print("Done!")

    # Extracting validation embeddings
    print("Extracting validation embeddings...")
    shape_embeds_val, key_to_shape_embeds_val, text_embeds_val, key_pair_to_text_embeds_val = extract_embeddings("val", data_loader_val)
    create_hdf5(args, "val", shape_embeds_val, key_to_shape_embeds_val, text_embeds_val, key_pair_to_text_embeds_val)
    print("Done!")

    print("Succesfully written to: [%s]" % args.data_path)


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)
