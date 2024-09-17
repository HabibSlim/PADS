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
        in_heads=1,
        dim_head=64,
        depth=2,
        weight_tie_layers=False,
    ):
        super().__init__()

        self.is_vae = False
        self.dim = dim
        self.latent_dim = latent_dim
        self.max_parts = max_parts
        self.heads = heads
        self.dim_head = dim_head
        self.depth = depth
        self.weight_tie_layers = weight_tie_layers

        cache_args = {"_cache": self.weight_tie_layers}

        # Centroid Embedding
        self.centroid_embed = PointEmbed(dim=dim // 2)
        # Label Embedding
        self.part_label_embed = nn.Embedding(COMPAT_FINE_PARTS, dim // 2)
        # Vector Embedding
        self.vector_embed = PointEmbed(dim=dim // 2)
        # Bounding Box Embeddings fusion
        self.bb_coord_proj_in = nn.Linear(2 * dim, dim // 2)
        self.bb_coord_proj_out = nn.Linear(2 * dim, dim)

        # Input/Output Cross-Attention Blocks
        self.in_encode = PreNorm(
            dim, Attention(dim, dim, heads=in_heads, dim_head=dim), context_dim=dim
        )
        self.in_decode = PreNorm(
            dim, Attention(dim, dim, heads=in_heads, dim_head=dim), context_dim=dim
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

    def get_part_embeds(self, part_bbs, part_labels, batch_mask):
        # Embed centroids of bounding boxes
        bb_centroids = part_bbs[:, :, 0, :]  # B x N_p x 3
        bb_centroid_embeds = self.centroid_embed(bb_centroids)  # B x N_p x D/2

        # Embed vectors of bounding boxes
        bb_vectors = part_bbs[:, :, 1:, :]  # B x N_p x 3 x 3
        bb_vectors = bb_vectors.reshape(-1, 24 * 3, 3)
        bb_vector_embeds = self.vector_embed(bb_vectors)  # B x N_p x D/2
        bb_vector_embeds = bb_vector_embeds.reshape(
            -1, 24, 3 * self.dim // 2
        )  # B x N_p x 3*D/2

        # Embed part labels (take mask into account)
        part_labels = part_labels * batch_mask
        part_labels_embed = self.part_label_embed(part_labels)  # B x N_p x D/2

        # Project the vector embeds + centroid embeds to 256
        bb_coord_embeds = torch.cat((bb_centroid_embeds, bb_vector_embeds), dim=-1)
        bb_coord_embeds_proj = self.bb_coord_proj_in(
            torch.cat((bb_centroid_embeds, bb_vector_embeds), dim=-1)
        )
        part_embeds = torch.cat(
            (bb_coord_embeds_proj, part_labels_embed), dim=-1
        )  # B x N_p x D

        return part_embeds, bb_coord_embeds

    def encode(self, latents, part_bbs, part_labels, batch_mask):
        # Get part labels/bbs embeddings
        part_embeds, bb_coord_embeds = self.get_part_embeds(
            part_bbs, part_labels, batch_mask
        )

        # Repeat latents to match the number of parts
        latents_in = latents.transpose(1, 2).repeat(1, 3, 1)  # B x D x 8 -> B x 24 x D
        x = self.in_encode(part_embeds, context=latents_in, mask=batch_mask)

        # Stacked encoder layers
        for attn, ff in self.encoder_layers:
            x = attn(x) + x
            x = ff(x) + x

        # Compress to desired reduced dimension
        part_latents = self.compress_latents(x)

        return part_latents, bb_coord_embeds

    def decode(self, part_latents, bb_coord_embeds, batch_mask):
        # Expand latents to full dimension
        x = self.expand_latents(part_latents)
        bb_coord_embeds_proj = self.bb_coord_proj_out(bb_coord_embeds)
        x = self.in_decode(x, context=bb_coord_embeds_proj, mask=batch_mask)

        # Stacked decoder layers
        for attn, ff in self.decoder_layers:
            x = attn(x) + x
            x = ff(x) + x

        # Apply output block
        latents = self.out_proj(x.transpose(1, 2))

        return latents

    def forward(self, latents, part_bbs, part_labels, batch_mask):
        part_latents, bb_coord_embeds = self.encode(
            latents, part_bbs, part_labels, batch_mask
        )
        logits = self.decode(part_latents, bb_coord_embeds, batch_mask)

        return logits, part_latents


class PartAwareVAE(PartAwareAE):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.is_vae = True
        self.mean_fc = nn.Linear(self.latent_dim, self.latent_dim)
        self.logvar_fc = nn.Linear(self.latent_dim, self.latent_dim)

    def get_part_embeds(self, part_bbs, part_labels, batch_mask):
        return super().get_part_embeds(part_bbs, part_labels, batch_mask)

    def encode(self, latents, part_bbs, part_labels, batch_mask, deterministic=False):
        part_latents, bb_coord_embeds = super().encode(
            latents, part_bbs, part_labels, batch_mask
        )

        mean = self.mean_fc(part_latents)
        logvar = self.logvar_fc(part_latents)

        posterior = DiagonalGaussianDistribution(
            mean, logvar, deterministic=deterministic, no_reduction=True
        )
        part_latents = posterior.sample()
        kl = posterior.kl()

        return kl, part_latents, bb_coord_embeds

    def decode(self, part_latents, bb_coord_embeds, batch_mask):
        latents = super().decode(part_latents, bb_coord_embeds, batch_mask)

        return latents

    def forward(self, latents, part_bbs, part_labels, batch_mask, deterministic=False):
        kl, part_latents, bb_coord_embeds = self.encode(
            latents, part_bbs, part_labels, batch_mask, deterministic=deterministic
        )
        logits = self.decode(part_latents, bb_coord_embeds, batch_mask).squeeze(-1)

        return logits, kl, part_latents

    @property
    def device(self):
        return next(self.parameters()).device
