"""
Transformer-based diffusion model for text-conditioned shape editing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import models.sampling as smp

from util.misc import default, zero_module
from models.attention import CrossAttention, FeedForward, MaskableCrossAttention
from models.modules import (
    AdaLayerNorm,
    LayerScale,
    StackedRandomGenerator,
    TimestepEmbedding,
)


class LatentArrayDenoiserBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        init_values=0,
        maskable_ca=False,
    ):
        super().__init__()
        self.attn1 = CrossAttention(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = MaskableCrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )  # is self-attn if context is none
        self.norm1 = AdaLayerNorm(dim)
        self.norm2 = AdaLayerNorm(dim)
        self.norm3 = AdaLayerNorm(dim)
        self.checkpoint = checkpoint

        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )

        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )

        self.ls3 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )

    def forward(self, x, t, context=None, context_mask=None):
        x = self.ls1(self.attn1(self.norm1(x, t))) + x
        x = (
            self.ls2(self.attn2(self.norm2(x, t), context=context, context_mask=None))
            + x
        )
        x = self.ls3(self.ff(self.norm3(x, t))) + x
        return x


class LatentArrayDenoiser(nn.Module):
    """
    Transformer block for denoising a latent array.
    """

    def __init__(
        self,
        in_channels,
        t_channels,
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

        self.t_channels = t_channels
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
                LatentArrayDenoiserBlock(
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
        self.map_noise = TimestepEmbedding(t_channels)

        self.map_layer0 = nn.Linear(in_features=t_channels, out_features=inner_dim)
        self.map_layer1 = nn.Linear(in_features=inner_dim, out_features=inner_dim)

    def forward(self, x, t, cond=None, cond_mask=None):
        t_emb = self.map_noise(t)[:, None]
        t_emb = F.silu(self.map_layer0(t_emb))
        t_emb = F.silu(self.map_layer1(t_emb))
        x = self.proj_in(x)

        for block in self.transformer_blocks:
            x = block(x, t_emb, context=cond, context_mask=cond_mask)

        x = self.norm(x)
        x = self.proj_out(x)
        return x


class EDMBase(torch.nn.Module):
    """
    Base class for the Latent Diffusion Model.
    """

    def __init__(
        self,
        n_latents=512,
        channels=8,
        use_fp16=False,
        sigma_min=0,
        sigma_max=float("inf"),
        sigma_data=1,
        n_heads=8,
        d_head=64,
        depth=12,
        out_channels=None,
    ):
        super().__init__()
        self.n_latents = n_latents
        self.channels = channels
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.out_channels = channels if out_channels is None else out_channels

        self.model = LatentArrayDenoiser(
            in_channels=channels,
            t_channels=256,
            n_heads=n_heads,
            d_head=d_head,
            depth=depth,
            out_channels=self.out_channels,
        )
        self.dtype = torch.float16 if use_fp16 else torch.float32

    def edm_forward(self, x, sigma, cond, cond_mask=None):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4
        c_noise = c_noise.to(self.dtype)

        F_x = self.model((c_in * x).to(self.dtype), c_noise.flatten(), cond, cond_mask)
        D_x = c_skip * x + c_out * F_x.to(self.dtype)

        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    @torch.no_grad()
    def sample(self, cond, latents=None, batch_seeds=None, num_steps=18, start_step=0):
        B, device = cond.shape[0], cond.device

        if latents is None:
            latents = torch.zeros([B, self.n_latents, self.channels], device=device)

        # Initialize the random generator
        if batch_seeds is None:
            batch_seeds = torch.arange(B)
        randn_like = StackedRandomGenerator(device, batch_seeds).randn_like

        return smp.edm_sampler(
            net=self,
            latents=latents,
            cond=cond,
            randn_like=randn_like,
            num_steps=num_steps,
            start_step=start_step,
        )

    def freeze_dm(self):
        """
        Freezes the latent array transformer by setting requires_grad to False for all parameters.
        """
        for param in self.model.parameters():
            param.requires_grad = False

    @property
    def device(self):
        return next(self.parameters()).device


class EDMPartQueries(EDMBase):
    def __init__(
        self,
        part_queries_encoder,
        n_latents=512,
        channels=8,
        use_fp16=False,
        sigma_min=0,
        sigma_max=float("inf"),
        sigma_data=1,
        n_heads=8,
        d_head=64,
        depth=12,
        out_channels=None,
    ):
        super().__init__(
            n_latents,
            channels,
            use_fp16,
            sigma_min,
            sigma_max,
            sigma_data,
            n_heads,
            d_head,
            depth,
            out_channels,
        )
        self.pqe = part_queries_encoder

    def forward(self, samples, sigma):
        # Unpack the data
        x, part_bbs, part_labels, batch_mask = samples

        x = x.to(self.dtype)
        part_bbs = part_bbs.to(self.dtype)
        sigma = sigma.to(self.dtype).reshape(-1, 1, 1)

        # Compute part queries condition
        cond, part_embeds = self.pqe(x, part_bbs, part_labels, batch_mask)

        return (
            self.edm_forward(x=x, sigma=sigma, cond=cond, cond_mask=batch_mask),
            part_embeds,
        )


class EDMPartAssets(EDMBase):
    def __init__(
        self,
        part_queries_encoder,
        n_latents=512,
        channels=8,
        use_fp16=False,
        sigma_min=0,
        sigma_max=float("inf"),
        sigma_data=1,
        n_heads=8,
        d_head=64,
        depth=12,
        out_channels=None,
    ):
        super().__init__(
            n_latents,
            channels,
            use_fp16,
            sigma_min,
            sigma_max,
            sigma_data,
            n_heads,
            d_head,
            depth,
            out_channels,
        )
        self.pqe = part_queries_encoder

    def forward(self, samples, sigma, num_samples=512):
        # Unpack the data
        x, part_bbs, part_labels, part_points, shape_cls, batch_mask = samples
        x = x.to(self.dtype)
        part_bbs = part_bbs.to(self.dtype)
        sigma = sigma.to(self.dtype).reshape(-1, 1, 1)

        # Compute part queries condition
        cond, part_embeds = self.pqe(
            part_bbs=part_bbs,
            part_points=part_points,
            batch_mask=batch_mask,
            shape_cls=shape_cls,
            num_samples=512,
        )

        return (
            self.edm_forward(x=x, sigma=sigma, cond=cond, cond_mask=batch_mask),
            part_embeds,
        )


class EDMPrecond(EDMBase):
    def __init__(
        self,
        n_latents=512,
        channels=8,
        use_fp16=False,
        sigma_min=0,
        sigma_max=float("inf"),
        sigma_data=1,
        n_heads=8,
        d_head=64,
        depth=12,
        out_channels=None,
        make_embeds=False,
    ):
        super().__init__(
            n_latents,
            channels,
            use_fp16,
            sigma_min,
            sigma_max,
            sigma_data,
            n_heads,
            d_head,
            depth,
            out_channels,
        )
        self.make_embeds = make_embeds
        if make_embeds:
            self.category_emb = nn.Embedding(55, n_heads * d_head)

    def emb_category(self, class_labels):
        return self.category_emb(class_labels).unsqueeze(1)

    def forward(self, x, sigma, cond=None, force_fp32=False, **model_kwargs):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1)
        if self.make_embeds:
            cond = self.category_emb(cond).unsqueeze(1)

        return self.edm_forward(x=x, sigma=sigma, cond=cond)

    @torch.no_grad()
    def sample(self, cond, latents=None, batch_seeds=None, num_steps=18, start_step=0):
        cond = self.category_emb(cond).unsqueeze(1)
        return super().sample(cond, latents, batch_seeds, num_steps, start_step)


def kl_d512_m512_l8_d24():
    model = EDMPrecond(n_latents=512, channels=8, depth=24, make_embeds=True)
    return model


def kl_d512_m512_l8_d24_pq(
    layer_depth=0, n_parts=24, part_latents_dim=512, single_learnable_query=False
):
    from models.part_queries import PQM

    pqm = PQM(
        dim=512,
        heads=8,
        dim_head=64,
    )
    model = EDMPartQueries(
        part_queries_encoder=pqm, n_latents=512, channels=8, depth=24
    )
    return model


def kl_d512_m512_l8_d24_passets(
    layer_depth=0, n_parts=24, single_learnable_query=False
):
    from models.points.encoders import pointbert_g512_d12_compat
    from models.part_assets import PartTokenizer

    pc_encoder = pointbert_g512_d12_compat()
    passets = PartTokenizer(
        pc_encoder=pc_encoder,
        bb_input_dim=12,
        bb_hidden_dim=64,
        bb_output_dim=32,
        bb_mlp_depth=3,
        visual_feature_dim=128,
        out_dim=512,
    )
    model = EDMPartAssets(
        part_queries_encoder=passets, n_latents=512, channels=8, depth=24
    )
    return model
