"""
Generating a set of part-aware latents from a set of part bounding boxes and part labels: "part queries".
"""

import torch
from torch import nn
from datasets.metadata import COMPAT_FINE_PARTS
from models.modules import (
    Attention,
    StackedAttentionBlocks,
    GEGLU,
    PointEmbed,
    PreNorm,
)


class PartEmbed(nn.Module):
    """
    Part-aware embeddings for part labels and bounding boxes.
    """

    def __init__(self, dim):
        super().__init__()

        self.embed_dim = dim

        # Embedding layers
        self.centroid_embed = PointEmbed(dim=dim // 2)
        self.vector_embed = PointEmbed(dim=dim // 2)
        self.part_label_embed = nn.Embedding(COMPAT_FINE_PARTS, dim // 2)

        # Projections
        self.bb_embeds_proj = nn.Sequential(
            nn.Linear(4 * dim // 2, dim // 2), nn.ReLU(), nn.Linear(dim // 2, dim // 2)
        )
        self.final_embeds_proj = nn.Linear(dim, dim)

        # Learnable empty object query
        self.empty_object_query = nn.Parameter(torch.randn(dim))

    def forward(self, part_bbs, part_labels, batch_mask):
        B, N, _, _ = part_bbs.shape
        D = self.embed_dim // 2

        # Embed centroids and vectors coordinates
        # ================================================
        bb_centroids = part_bbs[:, :, 0, :]
        bb_centroid_embeds = self.centroid_embed(bb_centroids)  # B x 24 x D

        bb_vectors = part_bbs[:, :, 1:, :].reshape(-1, N * 3, 3)
        bb_vector_embeds = self.vector_embed(bb_vectors)  # B x 72 x D

        # Interleave the embeddings
        bb_vector_embeds_reshaped = bb_vector_embeds.reshape(
            bb_centroid_embeds.shape[0], N, 3, -1
        )
        bb_embeds = torch.empty(
            (bb_centroid_embeds.shape[0], N * 4, bb_centroid_embeds.shape[2]),
            device=bb_centroid_embeds.device,
        )  # B x 96 x D
        bb_embeds[:, 0::4, :] = bb_centroid_embeds
        bb_embeds[:, 1::4, :] = bb_vector_embeds_reshaped[:, :, 0, :]
        bb_embeds[:, 2::4, :] = bb_vector_embeds_reshaped[:, :, 1, :]
        bb_embeds[:, 3::4, :] = bb_vector_embeds_reshaped[:, :, 2, :]

        # Project the embeddings (vectors + centroids) to the same dimension
        bb_embeds = bb_embeds.view(B, N, 4 * D)  # B x 24 x (D * 4)
        bb_embeds = self.bb_embeds_proj(bb_embeds)  # B x 24 x D
        # ================================================

        # Embed part labels
        # ================================================
        labels_embed = self.part_label_embed(part_labels * batch_mask)  # B x 24 x D
        # ================================================

        # Final embeddings
        # ================================================
        part_embeds = torch.cat([labels_embed, bb_embeds], dim=-1)
        part_embeds = self.final_embeds_proj(part_embeds)  # B x 24 x D

        # Replace masked (empty) objects with the learnable empty object query
        empty_mask = ~batch_mask.bool()
        part_embeds[empty_mask] = self.empty_object_query.expand(empty_mask.sum(), -1)
        # ================================================

        return part_embeds, labels_embed, bb_embeds


class PartQueriesGenerator(nn.Module):
    """
    Generating a set of part-aware latents
    from a set of part bounding boxes and part labels: "part queries".
    """

    def __init__(
        self,
        dim=512,
        latent_dim=128,
        max_parts=24,
        heads=8,
        in_heads=1,
        dim_head=64,
        depth=2,
        weight_tie_layers=False,
        use_attention_masking=True,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.max_parts = max_parts
        self.use_attention_masking = use_attention_masking

        # Part Embeddings
        self.part_embed = PartEmbed(dim)
        self.embed_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

        # Input Cross-Attention Block
        self.in_encode = PreNorm(
            dim, Attention(dim, dim, heads=in_heads, dim_head=dim), context_dim=dim
        )

        # Replace repeat with learnable projection
        self.latent_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

        # Stacked Attention Layers
        self.encoder_layers = nn.Identity()
        if depth > 0:
            self.encoder_layers = StackedAttentionBlocks(
                dim, depth, heads, dim_head, weight_tie_layers
            )

        # Compress latents to latent dimension
        self.compress_latents = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, latent_dim),
        )

    def forward(self, latents, part_bbs, part_labels, batch_mask):
        """
        :param latents:     B x 512 x 8
        :param part_bbs:    B x 24 x 4 x 3
        :param part_labels: B x 24
        :param batch_mask:  B x 24
        """
        # Embed part labels and bounding boxes
        part_embeds, labels_embed, bb_embeds = self.part_embed(
            part_bbs, part_labels, batch_mask
        )

        # Apply learnable projection instead of repeat
        latents_kv = latents.transpose(1, 2).repeat(1, 3, 1)  # B x 24 x 512
        latents_kv = self.latent_proj(latents_kv)

        # Concatenate part embeddings with latents
        part_embeds = self.embed_proj(part_embeds)  # B x 24 x 512

        # Encode part embeddings
        mask = batch_mask if self.use_attention_masking else None
        x = self.in_encode(part_embeds, context=latents_kv, mask=mask)  # B x 24 x 512
        x = self.encoder_layers(x)  # B x 24 x 512
        part_latents = self.compress_latents(x)  # B x 24 x 512
        return part_latents, part_embeds

    @property
    def device(self):
        return next(self.parameters()).device
