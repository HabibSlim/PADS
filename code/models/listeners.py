"""
Neural Listeners on shape pairs.
"""
import torch
from torch import nn
from models.mlp import MLP


class NeuralListener(torch.nn.Module):
    """
    Neural Listener returning a binary classification of the shape pair.
    Output two logits:
        1 if x_a is the edited version of x_b
        0 otherwise
    """

    def __init__(
        self,
        shape_latent_dim: int,
        text_embed_dim: int,
        bottleneck_dim: int = 256,
        encoder_depth: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.shape_encoder = MLP(
            shape_latent_dim, [bottleneck_dim, bottleneck_dim] * encoder_depth
        )
        self.proj_ft = nn.Linear(text_embed_dim, bottleneck_dim)
        self.predictor = MLP(
            bottleneck_dim * 3,
            [bottleneck_dim, bottleneck_dim // 2, bottleneck_dim // 4, 1],
            dropout_rate=dropout,
        )

    def forward(self, x_a, x_b, embed_ab):
        """
        x_a, x_b: input latents of the two shapes
        embed_ab: text embedding
        """
        x_a = self.shape_encoder(x_a)
        x_b = self.shape_encoder(x_b)
        embed_ab = self.proj_ft(embed_ab)

        x = torch.cat([x_a, x_b, embed_ab], dim=1)
        x = self.predictor(x)
        return x


class NeuralListenerShallow(torch.nn.Module):
    """
    Neural Listener returning a binary classification of the shape pair.
    Output two logits:
        1 if x_a is the edited version of x_b
        0 otherwise
    """

    def __init__(
        self,
        shape_latent_dim: int,
        text_embed_dim: int,
        bottleneck_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.shape_encoder = MLP(shape_latent_dim, [bottleneck_dim])
        self.proj_ft = nn.Linear(text_embed_dim, bottleneck_dim)
        self.predictor = MLP(
            bottleneck_dim * 3,
            [bottleneck_dim, bottleneck_dim // 2, 1],
            dropout_rate=dropout,
        )

    def forward(self, x_a, x_b, embed_ab):
        """
        x_a, x_b: input latents of the two shapes
        embed_ab: text embedding
        """
        x_a = self.shape_encoder(x_a)
        x_b = self.shape_encoder(x_b)
        embed_ab = self.proj_ft(embed_ab)

        x = torch.cat([x_a, x_b, embed_ab], dim=1)
        x = self.predictor(x)
        return x


## Latent mappers for 4096-dim shape latent space
# ==================================================


def nrl_listener_bert_shallow_64(use_linear_proj=False):
    return NeuralListenerShallow(
        shape_latent_dim=512 * 8,
        text_embed_dim=768,
        bottleneck_dim=64,
        dropout=0.5,
    )


def nrl_listener_bert_128(use_linear_proj=False):
    return NeuralListener(
        shape_latent_dim=512 * 8,
        text_embed_dim=768,
        bottleneck_dim=128,
        dropout=0.3,
    )


def nrl_listener_bert_256(use_linear_proj=False):
    return NeuralListener(
        shape_latent_dim=512 * 8,
        text_embed_dim=768,
        bottleneck_dim=256,
        dropout=0.3,
    )


def nrl_listener_bert_512(use_linear_proj=False):
    return NeuralListener(
        shape_latent_dim=512 * 8,
        text_embed_dim=768,
        bottleneck_dim=512,
        dropout=0.2,
    )


def nrl_listener_bert_512_deep(use_linear_proj=False):
    return NeuralListener(
        shape_latent_dim=512 * 8,
        text_embed_dim=768,
        bottleneck_dim=512,
        encoder_depth=2,
        dropout=0.2,
    )


## Neural listeners for 256-dim shape latent space
# ==================================================


def nrl_listener_bert_256_pcae(use_linear_proj=False):
    return NeuralListener(
        shape_latent_dim=256,
        text_embed_dim=768,
        bottleneck_dim=512,
        dropout=0.1,
    )


def nrl_listener_bert_512_pcae(use_linear_proj=False):
    return NeuralListener(
        shape_latent_dim=256,
        text_embed_dim=768,
        bottleneck_dim=512,
        dropout=0.1,
    )


def nrl_listener_bert_1024_pcae(use_linear_proj=False):
    return NeuralListener(
        shape_latent_dim=256,
        text_embed_dim=768,
        bottleneck_dim=1024,
        dropout=0.1,
    )


def nrl_listener_bert_512_deep_pcae(use_linear_proj=None):
    return NeuralListener(
        shape_latent_dim=256,
        text_embed_dim=768,
        bottleneck_dim=512,
        encoder_depth=2,
        dropout=0.1,
    )
