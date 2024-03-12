"""
Transformer-based latent mapper for text-conditioned shape editing.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(
            start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device
        )
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads

        if context_dim is None:
            context_dim = query_dim

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)

        if context is None:
            context = x

        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = torch.einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        if dim_out is None:
            dim_out = dim

        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(
            project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
    ):
        super().__init__()
        self.attn1 = CrossAttention(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )  # is self-attn if context is none

        init_values = 0
        drop_path = 0.0

        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.ls3 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path3 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, context=None):
        x = self.ls1(x) + self.drop_path1(self.attn1(x))
        x = self.ls2(x) + self.drop_path2(self.ff(x))
        x = self.ls3(x) + self.drop_path3(self.attn2(x, context=context))
        return x


class LatentArrayTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
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
    ):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head

        self.proj_in = nn.Linear(in_channels, inner_dim, bias=False)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
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

    def forward(self, x, cond=None):
        x = self.proj_in(x)
        for block in self.transformer_blocks:
            x = block(x, context=cond)
        x = self.norm(x)
        x = self.proj_out(x)
        return x


class TransformerLatentMapper(torch.nn.Module):
    def __init__(
        self,
        n_latents=512,
        channels=8,
        use_fp16=False,
        n_heads=8,
        d_head=64,
        depth=12,
        out_channels=None,
        use_linear_proj=False,
    ):
        super().__init__()
        self.n_latents = n_latents
        self.channels = channels
        self.use_fp16 = use_fp16
        self.out_channels = channels if out_channels is None else out_channels

        self.model = LatentArrayTransformer(
            in_channels=channels,
            n_heads=n_heads,
            d_head=d_head,
            depth=depth,
            out_channels=self.out_channels,
        )

        self.use_linear_proj = use_linear_proj
        if use_linear_proj:
            self.linear_proj = nn.Linear(768, n_latents)

    def forward(self, x_a, embeds_ab):
        """
        Forward pass of the model.
        """
        if self.use_linear_proj:
            # Reduce to 512 dim using a linear layer
            embeds_ab = self.linear_proj(embeds_ab)
        embeds_ab = embeds_ab.unsqueeze(1)

        x = self.model(x=x_a, cond=embeds_ab)
        return x


def tfm_mapper_bert_direct_d12(use_linear_proj):
    return TransformerLatentMapper(
        n_heads=8,
        d_head=64,
        depth=12,
        channels=8,
        use_linear_proj=use_linear_proj,
    )


def tfm_mapper_bert_direct_d8(use_linear_proj):
    return TransformerLatentMapper(
        n_heads=8,
        d_head=64,
        depth=8,
        channels=8,
        use_linear_proj=use_linear_proj,
    )


def tfm_mapper_bert_direct_d4(use_linear_proj):
    return TransformerLatentMapper(
        n_heads=8,
        d_head=64,
        depth=4,
        channels=8,
        use_linear_proj=use_linear_proj,
    )
