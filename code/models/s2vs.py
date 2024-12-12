"""
Autoencoder models for point clouds.
"""

from torch import nn
from models.modules import (
    DiagonalGaussianDistribution,
    PointEmbed,
)
from models.attention import (
    Attention,
    FeedForward,
    PreNorm,
)
from util.misc import cache_fn, fps_subsample


class AutoEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth=24,
        dim=512,
        queries_dim=512,
        output_dim=1,
        num_inputs=2048,
        num_latents=512,
        heads=8,
        dim_head=64,
        weight_tie_layers=False,
        decoder_ff=False,
    ):
        super().__init__()

        self.depth = depth

        self.num_inputs = num_inputs
        self.num_latents = num_latents

        self.cross_attend_blocks = nn.ModuleList(
            [
                PreNorm(
                    dim, Attention(dim, dim, heads=1, dim_head=dim), context_dim=dim
                ),
                PreNorm(dim, FeedForward(dim)),
            ]
        )

        self.point_embed = PointEmbed(dim=dim)

        get_latent_attn = lambda: PreNorm(
            dim, Attention(dim, heads=heads, dim_head=dim_head, drop_path_rate=0.1)
        )
        get_latent_ff = lambda: PreNorm(dim, FeedForward(dim, drop_path_rate=0.1))
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
            Attention(queries_dim, dim, heads=1, dim_head=dim),
            context_dim=dim,
        )
        self.decoder_ff = (
            PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None
        )

        self.to_outputs = (
            nn.Linear(queries_dim, output_dim)
            if output_dim is not None
            else nn.Identity()
        )

    def encode(self, pc):
        # pc: B x N x 3
        B, N, D = pc.shape
        assert N == self.num_inputs

        ratio = 1.0 * self.num_latents / self.num_inputs
        sampled_pc = fps_subsample(pc, ratio)

        sampled_pc_embeddings = self.point_embed(sampled_pc)
        pc_embeddings = self.point_embed(pc)

        cross_attn, cross_ff = self.cross_attend_blocks

        x = (
            cross_attn(sampled_pc_embeddings, context=pc_embeddings, mask=None)
            + sampled_pc_embeddings
        )
        x = cross_ff(x) + x

        return x

    def decode(self, x, queries):
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

    def forward(self, pc, queries):
        x = self.encode(pc)
        o = self.decode(x, queries).squeeze(-1)

        return {"logits": o}


class KLAutoEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth=24,
        dim=512,
        queries_dim=512,
        output_dim=1,
        num_inputs=2048,
        num_latents=512,
        latent_dim=64,
        heads=8,
        dim_head=64,
        weight_tie_layers=False,
        decoder_ff=False,
    ):
        super().__init__()

        self.depth = depth

        self.num_inputs = num_inputs
        self.num_latents = num_latents

        self.cross_attend_blocks = nn.ModuleList(
            [
                PreNorm(
                    dim, Attention(dim, dim, heads=1, dim_head=dim), context_dim=dim
                ),
                PreNorm(dim, FeedForward(dim)),
            ]
        )

        self.point_embed = PointEmbed(dim=dim)

        get_latent_attn = lambda: PreNorm(
            dim, Attention(dim, heads=heads, dim_head=dim_head, drop_path_rate=0.1)
        )
        get_latent_ff = lambda: PreNorm(dim, FeedForward(dim, drop_path_rate=0.1))
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
            Attention(queries_dim, dim, heads=1, dim_head=dim),
            context_dim=dim,
        )
        self.decoder_ff = (
            PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None
        )

        self.to_outputs = (
            nn.Linear(queries_dim, output_dim)
            if output_dim is not None
            else nn.Identity()
        )

        self.proj = nn.Linear(latent_dim, dim)

        self.mean_fc = nn.Linear(dim, latent_dim)
        self.logvar_fc = nn.Linear(dim, latent_dim)

    @property
    def device(self):
        return next(self.parameters()).device

    def encode(self, pc):
        # pc: B x N x 3
        B, N, D = pc.shape
        assert N == self.num_inputs

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

        return kl, x

    def decode(self, x, queries):
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

    def forward(self, pc, queries):
        kl, x = self.encode(pc)

        o = self.decode(x, queries).squeeze(-1)

        # return o.squeeze(-1), kl
        return {"logits": o, "kl": kl}


def create_autoencoder(dim=512, M=512, latent_dim=64, N=2048, determinisitc=False):
    if determinisitc:
        model = AutoEncoder(
            depth=24,
            dim=dim,
            queries_dim=dim,
            output_dim=1,
            num_inputs=N,
            num_latents=M,
            heads=8,
            dim_head=64,
        )
    else:
        model = KLAutoEncoder(
            depth=24,
            dim=dim,
            queries_dim=dim,
            output_dim=1,
            num_inputs=N,
            num_latents=M,
            latent_dim=latent_dim,
            heads=8,
            dim_head=64,
        )
    return model


def kl_d512_m512_l512(N=2048):
    return create_autoencoder(dim=512, M=512, latent_dim=512, N=N, determinisitc=False)


def kl_d512_m512_l64(N=2048):
    return create_autoencoder(dim=512, M=512, latent_dim=64, N=N, determinisitc=False)


def kl_d512_m512_l32(N=2048):
    return create_autoencoder(dim=512, M=512, latent_dim=32, N=N, determinisitc=False)


def kl_d512_m512_l16(N=2048):
    return create_autoencoder(dim=512, M=512, latent_dim=16, N=N, determinisitc=False)


def kl_d512_m512_l8(N=2048):
    return create_autoencoder(dim=512, M=512, latent_dim=8, N=N, determinisitc=False)


def kl_d512_m512_l4(N=2048):
    return create_autoencoder(dim=512, M=512, latent_dim=4, N=N, determinisitc=False)


def kl_d512_m512_l2(N=2048):
    return create_autoencoder(dim=512, M=512, latent_dim=2, N=N, determinisitc=False)


def kl_d512_m512_l1(N=2048):
    return create_autoencoder(dim=512, M=512, latent_dim=1, N=N, determinisitc=False)


###
def ae_d512_m512(N=2048):
    return create_autoencoder(dim=512, M=512, N=N, determinisitc=True)


def ae_d512_m256(N=2048):
    return create_autoencoder(dim=512, M=256, N=N, determinisitc=True)


def ae_d512_m128(N=2048):
    return create_autoencoder(dim=512, M=128, N=N, determinisitc=True)


def ae_d512_m64(N=2048):
    return create_autoencoder(dim=512, M=64, N=N, determinisitc=True)


###
def ae_d256_m512(N=2048):
    return create_autoencoder(dim=256, M=512, N=N, determinisitc=True)


def ae_d128_m512(N=2048):
    return create_autoencoder(dim=128, M=512, N=N, determinisitc=True)


def ae_d64_m512(N=2048):
    return create_autoencoder(dim=64, M=512, N=N, determinisitc=True)
