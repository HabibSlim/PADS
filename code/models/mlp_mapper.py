"""
MLP-based latent mapper for text-conditioned shape editing.
"""
import torch
import torch.nn.functional as F
from torch import nn
from models.mlp import MLP


class MLPLatentMapper(torch.nn.Module):
    """
    MLP Latent Mapper.
    """

    def __init__(
        self,
        shape_latent_dim: int,
        text_embed_dim: int,
        num_layers: int,
        dropout: float = 0.2,
        remove_dir_bias: bool = True,
    ):
        super().__init__()
        self.shape_latent_dim = shape_latent_dim
        self.dir_mlp = MLP(
            in_feat_dims=shape_latent_dim + text_embed_dim,
            out_channels=[shape_latent_dim] * num_layers,
            dropout_rate=dropout,
            b_norm=True,
            remove_final_bias=remove_dir_bias,
        )
        self.mag_mlp = MLP(
            in_feat_dims=shape_latent_dim + text_embed_dim,
            out_channels=[256, 128, 64, 1],
            dropout_rate=dropout,
            b_norm=True,
            remove_final_bias=True,
        )

    def forward(self, x_a, embed_ab):
        """
        x_a: input latent vector
        embed_ab: text embedding
        """
        x = torch.cat([x_a, embed_ab], dim=1)

        # Residual MLP
        dir_vec = self.dir_mlp(x)
        dir_vec = F.normalize(dir_vec, dim=1)

        # Residual magnitude MLP
        magnitude = self.mag_mlp(x)

        return dir_vec * magnitude

    def forward_decoupled(self, x_a, embed_ab):
        """
        x_a: input latent vector
        embed_ab: text embedding
        """
        x = torch.cat([x_a, embed_ab], dim=1)

        # Residual MLP
        dir_vec = self.dir_mlp(x)
        dir_vec = F.normalize(dir_vec, dim=1)

        # Residual magnitude MLP
        magnitude = self.mag_mlp(x)

        return dir_vec, magnitude


class MLPLatentMapperResidual(torch.nn.Module):
    """
    MLP Latent Mapper w/ residual connections.
    """

    def __init__(
        self,
        shape_latent_dim: int,
        text_embed_dim: int,
        num_layers: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.shape_latent_dim = shape_latent_dim

        self.mlps = []
        self.mlps += [
            MLP(
                in_feat_dims=shape_latent_dim + text_embed_dim,
                out_channels=[shape_latent_dim],
                dropout_rate=dropout,
                b_norm=True,
                remove_final_bias=False,
            )
        ]

        for _ in range(num_layers - 1):
            self.mlps += [
                MLP(
                    in_feat_dims=shape_latent_dim,
                    out_channels=[shape_latent_dim],
                    dropout_rate=dropout,
                    b_norm=True,
                    remove_final_bias=False,
                )
            ]

        self.mlps += [
            MLP(
                in_feat_dims=shape_latent_dim,
                out_channels=[shape_latent_dim],
                dropout_rate=dropout,
                b_norm=True,
                remove_final_bias=True,
            )
        ]

        # Register all MLPs as submodules
        for k, mlp in enumerate(self.mlps):
            self.add_module("mlp_{}".format(k), mlp)

        self.mag_mlp = MLP(
            in_feat_dims=shape_latent_dim + text_embed_dim,
            out_channels=[256, 128, 64, 1],
            dropout_rate=dropout,
            b_norm=True,
            remove_final_bias=True,
        )

    def forward(self, x_a, embed_ab):
        """
        x_a: input latent vector
        embed_ab: text embedding
        """
        # Residual layers
        x_0 = torch.cat([x_a, embed_ab], dim=1)
        x = self.mlps[0](x_0)
        last_x = x
        for k, mlp in enumerate(self.mlps[1:]):
            if k % 2 == 0:
                x = mlp(x)
            else:
                x = last_x + mlp(x)
                last_x = x

        dir_vec = F.normalize(x, dim=1)

        # Residual magnitude MLP
        magnitude = self.mag_mlp(x_0)

        return dir_vec * magnitude


class MLPLatentMapperBottlenecked(torch.nn.Module):
    """
    MLP Latent Mapper w/ bottleneck.
    """

    def __init__(
        self,
        shape_latent_dim: int,
        text_embed_dim: int,
        bottleneck_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        in_dim = text_embed_dim + shape_latent_dim
        self.editor = MLP(
            in_feat_dims=in_dim,
            out_channels=[
                bottleneck_dim,
                shape_latent_dim,
                shape_latent_dim,
                shape_latent_dim,
            ],
            dropout_rate=dropout,
            b_norm=True,
            remove_final_bias=True,
        )
        self.shape_encoder = MLP(
            in_feat_dims=shape_latent_dim,
            out_channels=[shape_latent_dim, shape_latent_dim],
            dropout_rate=dropout,
        )
        self.mag_mlp = MLP(
            in_feat_dims=in_dim, out_channels=[256, 128, 64, 1], closure=nn.ReLU()
        )

    def forward(self, x_a, embed_ab):
        """
        x_a: input latent vector
        embed_ab: text embedding
        """
        shape_fts = self.shape_encoder(x_a)
        x = torch.cat([shape_fts, embed_ab], dim=1)

        # Get the edit vector
        edit_vec = self.editor(x)
        edit_vec = F.normalize(edit_vec, dim=1)

        # Residual magnitude MLP
        magnitude = self.mag_mlp(x)

        return edit_vec * magnitude

    def forward_decoupled(self, x_a, embed_ab):
        """
        x_a: input latent vector
        embed_ab: text embedding
        """
        shape_fts = self.shape_encoder(x_a)
        x = torch.cat([shape_fts, embed_ab], dim=1)

        # Get the edit vector
        edit_vec = self.editor(x)
        edit_vec = F.normalize(edit_vec, dim=1)

        # Residual magnitude MLP
        magnitude = self.mag_mlp(x)

        return edit_vec, magnitude


class MLPLatentMapperBottleneckedCoupled(torch.nn.Module):
    """
    MLP Latent Mapper w/ bottleneck, coupled vector prediction.
    """

    def __init__(
        self,
        shape_latent_dim: int,
        text_embed_dim: int,
        bottleneck_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        in_dim = text_embed_dim + shape_latent_dim
        self.editor = MLP(
            in_feat_dims=in_dim,
            out_channels=[
                bottleneck_dim,
                shape_latent_dim,
                shape_latent_dim,
                shape_latent_dim,
            ],
            dropout_rate=dropout,
            b_norm=True,
            remove_final_bias=False,
        )
        self.shape_encoder = MLP(
            in_feat_dims=shape_latent_dim,
            out_channels=[shape_latent_dim, shape_latent_dim],
            dropout_rate=dropout,
        )

    def forward(self, x_a, embed_ab):
        """
        x_a: input latent vector
        embed_ab: text embedding
        """
        shape_fts = self.shape_encoder(x_a)
        x = torch.cat([shape_fts, embed_ab], dim=1)

        # Get the edit vector
        edit_vec = self.editor(x)

        return edit_vec


class MLPLatentMapperCoupled(torch.nn.Module):
    """
    MLP Latent Mapper, coupled prediction.
    """

    def __init__(
        self,
        shape_latent_dim: int,
        text_embed_dim: int,
        num_layers: int,
        dropout: float = 0.2,
        remove_dir_bias: bool = True,
    ):
        super().__init__()
        self.shape_latent_dim = shape_latent_dim
        self.editor = MLP(
            in_feat_dims=shape_latent_dim + text_embed_dim,
            out_channels=[shape_latent_dim] * num_layers,
            dropout_rate=dropout,
            b_norm=True,
            remove_final_bias=False,
        )

    def forward(self, x_a, embed_ab):
        """
        x_a: input latent vector
        embed_ab: text embedding
        """
        x = torch.cat([x_a, embed_ab], dim=1)

        # Get the edit vector
        edit_vec = self.editor(x)

        return edit_vec


class MLPLatentMapperDirect(torch.nn.Module):
    """
    MLP Latent Mapper with direct latent prediction.
    """

    def __init__(
        self,
        shape_latent_dim: int,
        text_embed_dim: int,
        num_layers: int,
        dropout: float = 0.2,
        remove_dir_bias: bool = True,
    ):
        super().__init__()
        self.shape_latent_dim = shape_latent_dim
        self.dir_mlp = MLP(
            in_feat_dims=shape_latent_dim + text_embed_dim,
            out_channels=[shape_latent_dim] * num_layers,
            dropout_rate=dropout,
            b_norm=True,
            remove_final_bias=remove_dir_bias,
        )

    def forward(self, x_a, embed_ab):
        """
        x_a: input latent vector
        embed_ab: text embedding
        """
        x = torch.cat([x_a, embed_ab], dim=1)

        # Residual MLP
        pred_latent = self.dir_mlp(x)

        return pred_latent


## Latent mappers for 4096-dim shape latent space
# ==================================================


def mlp_mapper_bert_l1_bias(use_linear_proj=None):
    model = MLPLatentMapper(
        shape_latent_dim=512 * 8,
        text_embed_dim=768,
        num_layers=1,
        remove_dir_bias=False,
    )
    return model


def mlp_mapper_bert_l1(use_linear_proj=None):
    model = MLPLatentMapper(shape_latent_dim=512 * 8, text_embed_dim=768, num_layers=1)
    return model


def mlp_mapper_bert_l2_low_dropout(use_linear_proj=None):
    model = MLPLatentMapper(
        shape_latent_dim=512 * 8, text_embed_dim=768, num_layers=2, dropout=0.05
    )
    return model


def mlp_mapper_bert_l2(use_linear_proj=None):
    model = MLPLatentMapper(shape_latent_dim=512 * 8, text_embed_dim=768, num_layers=2)
    return model


def mlp_mapper_bert_l4(use_linear_proj=None):
    model = MLPLatentMapper(shape_latent_dim=512 * 8, text_embed_dim=768, num_layers=4)
    return model


def mlp_mapper_bert_l8(use_linear_proj=None):
    model = MLPLatentMapper(shape_latent_dim=512 * 8, text_embed_dim=768, num_layers=8)
    return model


def mlp_mapper_bert_l8_residual(use_linear_proj=None):
    model = MLPLatentMapperResidual(
        shape_latent_dim=512 * 8, text_embed_dim=768, num_layers=8
    )
    return model


def mlp_mapper_bert_l16_residual(use_linear_proj=None):
    model = MLPLatentMapperResidual(
        shape_latent_dim=512 * 8, text_embed_dim=768, num_layers=16
    )
    return model


def mlp_mapper_bert_bneck_512(use_linear_proj=None):
    model = MLPLatentMapperBottlenecked(
        shape_latent_dim=512 * 8, text_embed_dim=768, bottleneck_dim=512, dropout=0.1
    )
    return model


def mlp_mapper_bert_bneck_256(use_linear_proj=None):
    model = MLPLatentMapperBottlenecked(
        shape_latent_dim=512 * 8, text_embed_dim=768, bottleneck_dim=256, dropout=0.1
    )
    return model


## Latent mappers for 256-dim shape latent space
# DECOUPLED
# ==================================================


def mlp_mapper_bert_bneck_256_pcae(use_linear_proj=None):
    model = MLPLatentMapperBottlenecked(
        shape_latent_dim=256, text_embed_dim=768, bottleneck_dim=256, dropout=0.1
    )
    return model


def mlp_mapper_bert_bneck_512_pcae(use_linear_proj=None):
    model = MLPLatentMapperBottlenecked(
        shape_latent_dim=256, text_embed_dim=768, bottleneck_dim=512, dropout=0.1
    )
    return model


def mlp_mapper_bert_bneck_1024_pcae(use_linear_proj=None):
    model = MLPLatentMapperBottlenecked(
        shape_latent_dim=256, text_embed_dim=768, bottleneck_dim=1024, dropout=0.1
    )
    return model


def mlp_mapper_bert_l4_pcae(use_linear_proj=None):
    model = MLPLatentMapper(shape_latent_dim=256, text_embed_dim=768, num_layers=4)
    return model


def mlp_mapper_bert_l8_pcae(use_linear_proj=None):
    model = MLPLatentMapper(shape_latent_dim=256, text_embed_dim=768, num_layers=8)
    return model


def mlp_mapper_bert_l8_residual_pcae(use_linear_proj=None):
    model = MLPLatentMapperResidual(
        shape_latent_dim=256, text_embed_dim=768, num_layers=8
    )
    return model


## Latent mappers for 256-dim shape latent space
# COUPLED
# ==================================================


def mlp_mapper_bert_bneck_256_pcae_cpl(use_linear_proj=None):
    model = MLPLatentMapperBottleneckedCoupled(
        shape_latent_dim=256, text_embed_dim=768, bottleneck_dim=256, dropout=0.1
    )
    return model


def mlp_mapper_bert_bneck_512_pcae_cpl(use_linear_proj=None):
    model = MLPLatentMapperBottleneckedCoupled(
        shape_latent_dim=256, text_embed_dim=768, bottleneck_dim=512, dropout=0.1
    )
    return model


def mlp_mapper_bert_bneck_1024_pcae_cpl(use_linear_proj=None):
    model = MLPLatentMapperBottleneckedCoupled(
        shape_latent_dim=256, text_embed_dim=768, bottleneck_dim=1024, dropout=0.1
    )
    return model


def mlp_mapper_bert_l4_pcae_cpl(use_linear_proj=None):
    model = MLPLatentMapperCoupled(
        shape_latent_dim=256, text_embed_dim=768, num_layers=4
    )
    return model


def mlp_mapper_bert_l8_pcae_cpl(use_linear_proj=None):
    model = MLPLatentMapperCoupled(
        shape_latent_dim=256, text_embed_dim=768, num_layers=8
    )
    return model


## Latent mappers for 256-dim shape latent space
# COUPLED
# ==================================================


def mlp_mapper_bert_direct_latent_256(use_linear_proj=None):
    model = MLPLatentMapperDirect(
        shape_latent_dim=256,
        text_embed_dim=768,
        num_layers=2,
        remove_dir_bias=False,
    )
    return model


def mlp_mapper_bert_direct_latent_512(use_linear_proj=None):
    model = MLPLatentMapperDirect(
        shape_latent_dim=512,
        text_embed_dim=768,
        num_layers=2,
        remove_dir_bias=False,
    )
    return model


def mlp_mapper_bert_direct_latent_1024(use_linear_proj=None):
    model = MLPLatentMapperDirect(
        shape_latent_dim=1024,
        text_embed_dim=768,
        num_layers=2,
        remove_dir_bias=False,
    )
    return model


## Latent mappers for 256-dim shape latent space
# SINGLE LAYER
# ==================================================


def mlp_mapper_bert_l1_bias__256(use_linear_proj=None):
    model = MLPLatentMapper(
        shape_latent_dim=256,
        text_embed_dim=768,
        num_layers=1,
        remove_dir_bias=False,
    )
    return model


def mlp_mapper_bert_l1__256(use_linear_proj=None):
    model = MLPLatentMapper(shape_latent_dim=256, text_embed_dim=768, num_layers=1)
    return model


def mlp_mapper_bert_l2__256(use_linear_proj=None):
    model = MLPLatentMapper(shape_latent_dim=256, text_embed_dim=768, num_layers=2)
    return model
