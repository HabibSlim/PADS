import torch
from torch import nn
from models.modules import (
    Attention,
    DiagonalGaussianDistribution,
    GEGLU,
    PreNorm,
    StackedAttentionBlocks,
)
from models.partqueries import PartQueriesGenerator


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
        use_attention_masking=True,
    ):
        super().__init__()

        self.is_vae = False
        self.dim = dim
        self.latent_dim = latent_dim
        self.use_attention_masking = use_attention_masking

        # Encoder
        self.encoder = PartQueriesGenerator(
            dim=dim,
            latent_dim=latent_dim,
            max_parts=max_parts,
            heads=heads,
            in_heads=in_heads,
            dim_head=dim_head,
            depth=depth,
            weight_tie_layers=weight_tie_layers,
            use_attention_masking=use_attention_masking,
        )

        # Decoder components
        self.in_decode = PreNorm(
            dim, Attention(dim, dim, heads=in_heads, dim_head=dim), context_dim=dim
        )

        self.decode_proj = nn.Linear(dim * 2, dim)
        self.out_proj = nn.Linear(24, 8)

        # Stacked Attention Layers for decoder
        self.decoder_layers = StackedAttentionBlocks(
            dim, depth, heads, dim_head, weight_tie_layers
        )

        # Expand latents
        self.expand_latents = nn.Sequential(
            nn.Linear(latent_dim, dim),
            GEGLU(),
            nn.Linear(dim // 2, dim),
        )

    def decode(self, part_latents, bb_embeds, batch_mask):
        x = self.expand_latents(part_latents)
        mask = batch_mask if self.use_attention_masking else None
        x = torch.cat([x, bb_embeds], dim=-1)
        x = self.decode_proj(x)
        x = self.in_decode(x, mask=mask)  # , context=bb_embeds
        x = self.decoder_layers(x)
        x = self.out_proj(x.transpose(1, 2))
        return x

    def forward(self, latents, part_bbs, part_labels, batch_mask):
        part_latents, bb_embeds = self.encoder(
            latents, part_bbs, part_labels, batch_mask
        )
        logits = self.decode(part_latents, bb_embeds, batch_mask)

        return logits, part_latents

    @property
    def device(self):
        return next(self.parameters()).device


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

    def encode(self, latents, part_bbs, part_labels, batch_mask, deterministic=False):
        part_latents, bb_embeds = self.encoder(
            latents, part_bbs, part_labels, batch_mask
        )

        mean = self.mean_fc(part_latents)
        logvar = self.logvar_fc(part_latents)

        posterior = DiagonalGaussianDistribution(
            mean, logvar, deterministic=deterministic, no_reduction=True
        )
        part_latents = posterior.sample()
        kl = posterior.kl()

        return kl, part_latents, bb_embeds

    def forward(self, latents, part_bbs, part_labels, batch_mask, deterministic=False):
        kl, part_latents, bb_embeds = self.encode(
            latents, part_bbs, part_labels, batch_mask, deterministic=deterministic
        )
        logits = self.decode(part_latents, bb_embeds, batch_mask).squeeze(-1)

        return logits, kl, part_latents
