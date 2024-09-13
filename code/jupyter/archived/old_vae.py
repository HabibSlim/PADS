"""
Part-aware autoencoders.
"""

import torch
from torch import nn
from datasets.metadata import COMPAT_FINE_PARTS
from models.modules import (
    Attention,
    DiagonalGaussianDistribution,
    FeedForward,
    GEGLU,
    PointEmbed,
    PreNorm,
)
from util.misc import cache_fn


class PartAwareAE(nn.Module):
    def __init__(
        self,
        dim=512,
        latent_dim=128,
        max_parts=24,
        heads=8,
        dim_head=64,
        depth=2,
        weight_tie_layers=False,
    ):
        super().__init__()

        self.dim = dim
        self.latent_dim = latent_dim
        self.max_parts = max_parts
        self.heads = heads
        self.dim_head = dim_head
        self.depth = depth
        self.weight_tie_layers = weight_tie_layers

        cache_args = {"_cache": self.weight_tie_layers}

        # Point Embedding
        self.point_embed = PointEmbed(dim=dim // 2)
        # Label Embedding
        self.part_label_embed = nn.Embedding(COMPAT_FINE_PARTS, dim // 2)

        # Input/Output Cross-Attention Blocks
        self.in_block = PreNorm(
            dim, Attention(dim, dim, heads=1, dim_head=dim), context_dim=dim
        )
        self.out_proj = nn.Linear(24, 8)

        # Stacked Attention Layers
        def get_latent_attn():
            return PreNorm(
                dim, Attention(dim, heads=heads, dim_head=dim_head, drop_path_rate=0.1)
            )

        def get_latent_ff():
            return PreNorm(dim, FeedForward(dim, drop_path_rate=0.1))

        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.encoder_layers = nn.ModuleList([])
        for i in range(depth):
            self.encoder_layers.append(
                nn.ModuleList(
                    [get_latent_attn(**cache_args), get_latent_ff(**cache_args)]
                )
            )
        self.decoder_layers = nn.ModuleList([])
        for i in range(depth):
            self.decoder_layers.append(
                nn.ModuleList(
                    [get_latent_attn(**cache_args), get_latent_ff(**cache_args)]
                )
            )

        # Compress/Expand latents
        self.compress_latents = nn.Sequential(
            nn.Linear(dim, dim // 2),
            GEGLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.expand_latents = nn.Sequential(
            nn.Linear(latent_dim, dim),
            GEGLU(),
            nn.Linear(dim // 2, dim),
        )

    def encode(self, latents, part_bbs, part_labels):
        # Compute the mask from batch labels (part labels equal to -1 are masked)
        batch_mask = part_labels != -1
        batch_mask = batch_mask.to(latents.device)

        # Embed bounding boxes
        bb_centroids = torch.mean(part_bbs, dim=-2)  # B x 24 x 3
        bb_embeds = self.point_embed(bb_centroids)  # B x 24 x 256

        # Embed part labels (take mask into account)
        part_labels = part_labels * batch_mask
        part_labels_embed = self.part_label_embed(part_labels)  # B x 24 x 256

        # Repeat latents to match the number of parts
        latents_in = latents.transpose(1, 2).repeat(
            1, 3, 1
        )  # B x 512 x 8 -> B x 24 x 512
        part_embeds = torch.cat((bb_embeds, part_labels_embed), dim=-1)

        x = self.in_block(part_embeds, context=latents_in, mask=batch_mask)

        # Stacked encoder layers
        for attn, ff in self.encoder_layers:
            x = attn(x) + x
            x = ff(x) + x

        # Compress to desired reduced dimension
        part_latents = self.compress_latents(x)

        return part_latents

    def decode(self, part_latents):
        # Expand latents to full dimension
        x = self.expand_latents(part_latents)

        # Stacked decoder layers
        for attn, ff in self.decoder_layers:
            x = attn(x) + x
            x = ff(x) + x

        # Apply output block
        latents = self.out_proj(x.transpose(1, 2))

        return latents

    def forward(self, latents, part_bbs, part_labels):
        encoded = self.encode(latents, part_bbs, part_labels)
        latents = self.decode(encoded)
        return latents


class PartAwareVAE(PartAwareAE):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.mean_fc = nn.Linear(self.latent_dim, self.latent_dim)
        self.logvar_fc = nn.Linear(self.latent_dim, self.latent_dim)

    def encode(self, latents, part_bbs, part_labels):
        x = super().encode(latents, part_bbs, part_labels)

        mean = self.mean_fc(x)
        logvar = self.logvar_fc(x)

        posterior = DiagonalGaussianDistribution(mean, logvar)
        x = posterior.sample()
        kl = posterior.kl()

        return kl, x

    def decode(self, part_latents):
        return super().decode(part_latents)

    def forward(self, latents, part_bbs, part_labels):
        kl, part_latents = self.encode(latents, part_bbs, part_labels)
        logits = self.decode(part_latents).squeeze(-1)

        return logits, kl
