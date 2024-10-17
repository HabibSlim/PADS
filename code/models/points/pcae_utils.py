"""
PC-AE utility functions for loading the pretrained model.
"""

import argparse
import json
import os.path as osp

import torch
import warnings

from models.mlp import MLP
from models.point_net import PointNet
from models.pointcloud_autoencoder import PointcloudAutoencoder


def describe_pc_ae(args):
    # Make an AE.
    if args.encoder_net == "pointnet":
        ae_encoder = PointNet(init_feat_dim=3, conv_dims=args.encoder_conv_layers)
        encoder_latent_dim = args.encoder_conv_layers[-1]
    else:
        raise NotImplementedError()

    if args.decoder_net == "mlp":
        ae_decoder = MLP(
            in_feat_dims=encoder_latent_dim,
            out_channels=args.decoder_fc_neurons + [args.n_pc_points * 3],
            b_norm=False,
        )

    model = PointcloudAutoencoder(ae_encoder, ae_decoder)
    return model


def load_state_dicts(checkpoint_file, map_location=None, **kwargs):
    if map_location is None:
        checkpoint = torch.load(checkpoint_file)
    else:
        checkpoint = torch.load(checkpoint_file, map_location=map_location)

    for key, value in kwargs.items():
        value.load_state_dict(checkpoint[key])

    epoch = checkpoint.get("epoch")
    if epoch:
        return epoch


def read_saved_args(config_file, override_or_add_args=None, verbose=False):
    parser = argparse.ArgumentParser()
    args = parser.parse_args([])
    with open(config_file, "r") as f_in:
        args.__dict__ = json.load(f_in)

    if override_or_add_args is not None:
        for key, val in override_or_add_args.items():
            args.__setattr__(key, val)

    return args


def load_pretrained_pc_ae(model_file):
    config_file = osp.join(osp.dirname(model_file), "config.json.txt")
    pc_ae_args = read_saved_args(config_file)
    pc_ae = describe_pc_ae(pc_ae_args)

    if osp.join(pc_ae_args.log_dir, "best_model.pt") != osp.abspath(model_file):
        warnings.warn(
            "The saved best_model.pt in the corresponding log_dir is not equal to the one requested."
        )

    load_state_dicts(model_file, model=pc_ae)
    return pc_ae, pc_ae_args
