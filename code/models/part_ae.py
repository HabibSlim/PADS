"""
Part Autoencoder model.
"""

import torch
from torch import nn
from models.modules import (
    DiagonalGaussianDistribution,
    PointEmbed,
)
from models.attention import (
    Attention,
    LatentTransformerBlock,
    FeedForward,
    PreNorm,
)
from util.misc import cache_fn, fps_subsample, zero_module


class KLEncoder(nn.Module):
    """
    Encoder with KL divergence loss.
    """

    def __init__(
        self,
        *,
        embed_dim,
        num_inputs,
        num_latents,
        latent_dim,
        dim_head,  # Added dim_head parameter
    ):
        super().__init__()

        self.num_inputs = num_inputs
        self.num_latents = num_latents

        self.cross_attend_blocks = nn.ModuleList(
            [
                PreNorm(
                    embed_dim,
                    Attention(embed_dim, embed_dim, heads=1, dim_head=dim_head),
                    context_dim=embed_dim,
                ),
                PreNorm(embed_dim, FeedForward(embed_dim)),
            ]
        )

        self.point_embed = PointEmbed(dim=embed_dim)

        self.mean_fc = nn.Linear(embed_dim, latent_dim)
        self.logvar_fc = nn.Linear(embed_dim, latent_dim)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, pc):
        # pc: B x N x 3
        B, N, D = pc.shape

        ###### fps
        ratio = 1.0 * self.num_latents / self.num_inputs
        sampled_pc = fps_subsample(pc, ratio)
        ######

        sampled_pc_embeddings = self.point_embed(sampled_pc)
        pc_embeddings = self.point_embed(pc)

        cross_attn, cross_ff = self.cross_attend_blocks

        x = (
            cross_attn(sampled_pc_embeddings, context=pc_embeddings, mask=None)
            + sampled_pc_embeddings
        )
        x = cross_ff(x) + x

        mean = self.mean_fc(x)
        logvar = self.logvar_fc(x)

        posterior = DiagonalGaussianDistribution(mean, logvar)
        x = posterior.sample()
        kl = posterior.kl()

        return x, kl


class PartEncoder(nn.Module):
    """
    Encode every part pointcloud in parallel using the batch dimension.
    """

    def __init__(
        self,
        *,
        embed_dim=512,
        num_inputs=2048,
        num_latents=512,
        latent_dim=64,
        dim_head=64,
    ):
        super().__init__()

        self.encoder = KLEncoder(
            embed_dim=embed_dim,
            num_inputs=num_inputs,
            num_latents=num_latents,
            latent_dim=latent_dim,
            dim_head=dim_head,  # Added dim_head parameter
        )

    def forward(self, part_pcs):
        B, N, N_p, D = part_pcs.shape

        # Reshape to (B*N, N_p, D) to process all parts in parallel
        reshaped_pcs = part_pcs.view(B * N, N_p, D)
        x, kl = self.encoder(reshaped_pcs)

        output_shape = x.shape[1:]
        return x.view(B, N, *output_shape), kl


class ShapeDecoder(nn.Module):
    """
    Occupancy decoder taking a latent code and a query pointcloud as input.
    """

    def __init__(
        self,
        *,
        depth=24,
        embed_dim=512,
        queries_dim=512,
        output_dim=1,
        latent_dim=64,
        heads=8,
        dim_head=64,
        weight_tie_layers=False,
        decoder_ff=False,
    ):
        super().__init__()

        self.depth = depth

        get_latent_attn = lambda: PreNorm(
            embed_dim,
            Attention(embed_dim, heads=heads, dim_head=dim_head, drop_path_rate=0.1),
        )
        get_latent_ff = lambda: PreNorm(
            embed_dim, FeedForward(embed_dim, drop_path_rate=0.1)
        )
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {"_cache": weight_tie_layers}

        for i in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [get_latent_attn(**cache_args), get_latent_ff(**cache_args)]
                )
            )

        self.decoder_cross_attn = PreNorm(
            queries_dim,
            Attention(queries_dim, embed_dim, heads=1, dim_head=embed_dim),
            context_dim=embed_dim,
        )

        self.decoder_ff = (
            PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None
        )

        self.to_outputs = (
            nn.Linear(queries_dim, output_dim)
            if output_dim is not None
            else nn.Identity()
        )

        self.proj = nn.Linear(latent_dim, embed_dim)
        self.point_embed = PointEmbed(dim=queries_dim)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, queries):
        x = self.proj(x)

        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        # cross attend from decoder queries to latents
        queries_embeddings = self.point_embed(queries)
        latents = self.decoder_cross_attn(queries_embeddings, context=x)

        # optional decoder feedforward
        if self.decoder_ff is not None:
            latents = latents + self.decoder_ff(latents)

        return self.to_outputs(latents)


class BoundingBoxTokenizer(nn.Module):
    """
    Bounding box tokenizer using an MLP.
    """

    def __init__(
        self, bb_input_dim=24, mlp_hidden_dim=64, mlp_output_dim=32, mlp_depth=3
    ):
        super().__init__()
        self.mlp = self._build_mlp(
            bb_input_dim, mlp_hidden_dim, mlp_output_dim, mlp_depth
        )
        self.output_dim = mlp_output_dim

    def _build_mlp(self, input_dim, hidden_dim, output_dim, depth):
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(depth - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        return nn.Sequential(*layers)

    def forward(self, part_bbs):
        B, P, _, _ = part_bbs.shape  # B, P, bb_dim, 3

        # Flatten the bounding boxes
        flattened_bbs = part_bbs.view(B, P, -1)  # B, P, bb_dim * 3

        # Process through MLP
        pose_tokens = self.mlp(flattened_bbs)  # B, P, mlp_output_dim

        return pose_tokens


class LatentArrayTransformer(nn.Module):
    """
    Latent array transformer.
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        out_channels=None,
        deep_proj=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head

        if deep_proj:
            self.proj_in = nn.Sequential(
                nn.Linear(in_channels, n_heads * d_head),
                nn.GELU(),
                nn.Linear(n_heads * d_head, n_heads * d_head),
                nn.GELU(),
            )
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim, bias=False)

        self.transformer_blocks = nn.ModuleList(
            [
                LatentTransformerBlock(
                    inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(inner_dim)

        if out_channels is None:
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels, bias=False))
        else:
            self.proj_out = zero_module(nn.Linear(inner_dim, out_channels, bias=False))

        self.context_dim = context_dim

    def forward(self, x, cond=None, cond_mask=None):
        x = self.proj_in(x)

        for block in self.transformer_blocks:
            x = block(x, context=cond, context_mask=cond_mask)

        x = self.norm(x)
        x = self.proj_out(x)
        return x


class PartAE(nn.Module):
    """
    Part auto-encoder supporting bounding box supervision.
    """

    def __init__(
        self,
        *,
        max_parts=16,
        decoder_depth=24,
        mixer_depth=8,
        embed_dim=512,
        queries_dim=512,
        output_dim=1,
        num_inputs=2048,
        part_latent_dim=128,
        heads=8,
        dim_head=64,
        weight_tie_layers=False,
        decoder_ff=False,
    ):
        super().__init__()

        num_latents = part_latent_dim // 8
        latent_dim = 8

        self.encoder = PartEncoder(
            embed_dim=embed_dim,
            num_inputs=num_inputs,
            num_latents=num_latents,
            latent_dim=latent_dim,
            dim_head=dim_head,
        )

        self.bb_tokenizer = BoundingBoxTokenizer(
            bb_input_dim=24,
            mlp_hidden_dim=64,
            mlp_output_dim=part_latent_dim,
            mlp_depth=3,
        )

        self.part_mixer = LatentArrayTransformer(
            in_channels=part_latent_dim * 2,
            out_channels=part_latent_dim,
            n_heads=heads,
            d_head=dim_head,
            depth=mixer_depth,
            deep_proj=False,
        )

        self.decoder = ShapeDecoder(
            depth=decoder_depth,
            embed_dim=embed_dim,
            queries_dim=queries_dim,
            output_dim=output_dim,
            latent_dim=part_latent_dim,
            heads=heads,
            dim_head=dim_head,
            weight_tie_layers=weight_tie_layers,
            decoder_ff=decoder_ff,
        )

        self.part_latents_proj = nn.Linear(part_latent_dim, part_latent_dim)
        self.max_parts = max_parts

    def forward(self, part_points, part_bbs, queries):
        """
        B: Batch size
        N: Number of parts
        N_p: Number of points per part
        D: Point dimension == 3
        """
        B, N, N_p, D = part_points.shape
        B, N, K, D = part_bbs.shape

        # Forward all parts through the autoencoder at once
        encoded_parts, kl = self.encoder(part_points)  # B x N x part_latent_dim
        bb_tokens = self.bb_tokenizer(part_bbs)  # B x N x part_latent_dim

        # Concatenate the part latent codes with the bounding box tokens channel-wise
        encoded_parts = encoded_parts.view(B, N, -1)
        encoded_parts = self.part_latents_proj(
            encoded_parts
        )  # TODO: Maybe some issues here related to permutation invariance
        decoder_input = torch.cat([encoded_parts, bb_tokens], dim=-1)

        # Mix the part latent codes and bounding box tokens
        mixed_parts = self.part_mixer(decoder_input)

        encoded_parts, kl = self.encoder(
            part_points
        )  # B x N x num_latents * latent_dim

        return {"logits": self.decoder(mixed_parts, queries), "kl": kl}

    @property
    def device(self):
        return next(self.parameters()).device


def part_ae_d8_m8_l32():
    """
    decoder_depth=8, mixer_depth=8, part_latent_dim=32
    """
    return PartAE(
        max_parts=16,
        decoder_depth=8,
        mixer_depth=8,
        embed_dim=512,
        queries_dim=512,
        output_dim=1,
        num_inputs=1024,
        part_latent_dim=32,
        heads=8,
        dim_head=64,
        weight_tie_layers=False,
        decoder_ff=False,
    )


def part_ae_d8_m8_l64():
    """
    decoder_depth=8, mixer_depth=8, part_latent_dim=64
    """
    return PartAE(
        max_parts=16,
        decoder_depth=8,
        mixer_depth=8,
        embed_dim=512,
        queries_dim=512,
        output_dim=1,
        num_inputs=1024,
        part_latent_dim=64,
        heads=8,
        dim_head=64,
        weight_tie_layers=False,
        decoder_ff=False,
    )


def part_ae_d8_m8_l128():
    """
    decoder_depth=8, mixer_depth=8, part_latent_dim=128
    """
    return PartAE(
        max_parts=16,
        decoder_depth=8,
        mixer_depth=8,
        embed_dim=512,
        queries_dim=512,
        output_dim=1,
        num_inputs=1024,
        part_latent_dim=128,
        heads=8,
        dim_head=64,
        weight_tie_layers=False,
        decoder_ff=False,
    )


def part_ae_d8_m16_l32():
    """
    decoder_depth=8, mixer_depth=16, part_latent_dim=32
    """
    return PartAE(
        max_parts=16,
        decoder_depth=8,
        mixer_depth=16,
        embed_dim=512,
        queries_dim=512,
        output_dim=1,
        num_inputs=1024,
        part_latent_dim=32,
        heads=8,
        dim_head=64,
        weight_tie_layers=False,
        decoder_ff=False,
    )


def part_ae_d8_m16_l64():
    """
    decoder_depth=8, mixer_depth=16, part_latent_dim=64
    """
    return PartAE(
        max_parts=16,
        decoder_depth=8,
        mixer_depth=16,
        embed_dim=512,
        queries_dim=512,
        output_dim=1,
        num_inputs=1024,
        part_latent_dim=64,
        heads=8,
        dim_head=64,
        weight_tie_layers=False,
        decoder_ff=False,
    )


def part_ae_d8_m16_l128():
    """
    decoder_depth=8, mixer_depth=16, part_latent_dim=128
    """
    return PartAE(
        max_parts=16,
        decoder_depth=8,
        mixer_depth=16,
        embed_dim=512,
        queries_dim=512,
        output_dim=1,
        num_inputs=1024,
        part_latent_dim=128,
        heads=8,
        dim_head=64,
        weight_tie_layers=False,
        decoder_ff=False,
    )


def part_ae_d16_m8_l32():
    """
    decoder_depth=16, mixer_depth=8, part_latent_dim=32
    """
    return PartAE(
        max_parts=16,
        decoder_depth=16,
        mixer_depth=8,
        embed_dim=512,
        queries_dim=512,
        output_dim=1,
        num_inputs=1024,
        part_latent_dim=32,
        heads=8,
        dim_head=64,
        weight_tie_layers=False,
        decoder_ff=False,
    )


def part_ae_d16_m8_l64():
    """
    decoder_depth=16, mixer_depth=8, part_latent_dim=64
    """
    return PartAE(
        max_parts=16,
        decoder_depth=16,
        mixer_depth=8,
        embed_dim=512,
        queries_dim=512,
        output_dim=1,
        num_inputs=1024,
        part_latent_dim=64,
        heads=8,
        dim_head=64,
        weight_tie_layers=False,
        decoder_ff=False,
    )


def part_ae_d16_m8_l128():
    """
    decoder_depth=16, mixer_depth=8, part_latent_dim=128
    """
    return PartAE(
        max_parts=16,
        decoder_depth=16,
        mixer_depth=8,
        embed_dim=512,
        queries_dim=512,
        output_dim=1,
        num_inputs=1024,
        part_latent_dim=128,
        heads=8,
        dim_head=64,
        weight_tie_layers=False,
        decoder_ff=False,
    )


def part_ae_d16_m16_l32():
    """
    decoder_depth=16, mixer_depth=16, part_latent_dim=32
    """
    return PartAE(
        max_parts=16,
        decoder_depth=16,
        mixer_depth=16,
        embed_dim=512,
        queries_dim=512,
        output_dim=1,
        num_inputs=1024,
        part_latent_dim=32,
        heads=8,
        dim_head=64,
        weight_tie_layers=False,
        decoder_ff=False,
    )


def part_ae_d16_m16_l64():
    """
    decoder_depth=16, mixer_depth=16, part_latent_dim=64
    """
    return PartAE(
        max_parts=16,
        decoder_depth=16,
        mixer_depth=16,
        embed_dim=512,
        queries_dim=512,
        output_dim=1,
        num_inputs=1024,
        part_latent_dim=64,
        heads=8,
        dim_head=64,
        weight_tie_layers=False,
        decoder_ff=False,
    )


def part_ae_d16_m16_l128():
    """
    decoder_depth=16, mixer_depth=16, part_latent_dim=128
    """
    return PartAE(
        max_parts=16,
        decoder_depth=16,
        mixer_depth=16,
        embed_dim=512,
        queries_dim=512,
        output_dim=1,
        num_inputs=1024,
        part_latent_dim=128,
        heads=8,
        dim_head=64,
        weight_tie_layers=False,
        decoder_ff=False,
    )


def part_ae_d24_m8_l32():
    """
    decoder_depth=24, mixer_depth=8, part_latent_dim=32
    """
    return PartAE(
        max_parts=16,
        decoder_depth=24,
        mixer_depth=8,
        embed_dim=512,
        queries_dim=512,
        output_dim=1,
        num_inputs=1024,
        part_latent_dim=32,
        heads=8,
        dim_head=64,
        weight_tie_layers=False,
        decoder_ff=False,
    )


def part_ae_d24_m8_l64():
    """
    decoder_depth=24, mixer_depth=8, part_latent_dim=64
    """
    return PartAE(
        max_parts=16,
        decoder_depth=24,
        mixer_depth=8,
        embed_dim=512,
        queries_dim=512,
        output_dim=1,
        num_inputs=1024,
        part_latent_dim=64,
        heads=8,
        dim_head=64,
        weight_tie_layers=False,
        decoder_ff=False,
    )


def part_ae_d24_m8_l128():
    """
    decoder_depth=24, mixer_depth=8, part_latent_dim=128
    """
    return PartAE(
        max_parts=16,
        decoder_depth=24,
        mixer_depth=8,
        embed_dim=512,
        queries_dim=512,
        output_dim=1,
        num_inputs=1024,
        part_latent_dim=128,
        heads=8,
        dim_head=64,
        weight_tie_layers=False,
        decoder_ff=False,
    )


def part_ae_d24_m16_l32():
    """
    decoder_depth=24, mixer_depth=16, part_latent_dim=32
    """
    return PartAE(
        max_parts=16,
        decoder_depth=24,
        mixer_depth=16,
        embed_dim=512,
        queries_dim=512,
        output_dim=1,
        num_inputs=1024,
        part_latent_dim=32,
        heads=8,
        dim_head=64,
        weight_tie_layers=False,
        decoder_ff=False,
    )


def part_ae_d24_m16_l64():
    """
    decoder_depth=24, mixer_depth=16, part_latent_dim=64
    """
    return PartAE(
        max_parts=16,
        decoder_depth=24,
        mixer_depth=16,
        embed_dim=512,
        queries_dim=512,
        output_dim=1,
        num_inputs=1024,
        part_latent_dim=64,
        heads=8,
        dim_head=64,
        weight_tie_layers=False,
        decoder_ff=False,
    )


def part_ae_d24_m16_l128():
    """
    decoder_depth=24, mixer_depth=16, part_latent_dim=128
    """
    return PartAE(
        max_parts=16,
        decoder_depth=24,
        mixer_depth=16,
        embed_dim=512,
        queries_dim=512,
        output_dim=1,
        num_inputs=1024,
        part_latent_dim=128,
        heads=8,
        dim_head=64,
        weight_tie_layers=False,
        decoder_ff=False,
    )
