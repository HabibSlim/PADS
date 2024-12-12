"""
Attention-related modules.
"""

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath
from torch import einsum, nn
from util.misc import default, zero_module


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class GEGLUProj(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    """
    Feed-forward module with optional drop path.
    """

    def __init__(
        self, dim, dim_out=None, mult=4, glu=False, dropout=0.0, drop_path_rate=0.0
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        if dim_out is None:
            dim_out = dim

        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLUProj(dim, inner_dim)
        )

        self.net = nn.Sequential(
            project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

        if drop_path_rate > 0.0:
            assert dropout == 0.0, "dropout should be 0 when using drop_path"

    def forward(self, x):
        return self.drop_path(self.net(x))


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

    def forward(self, x, context=None):
        h = self.heads
        q = self.to_q(x)

        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

        attn = sim.softmax(dim=-1)
        out = torch.einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class MaskableCrossAttention(nn.Module):
    """
    Maskable cross-attention module.
    """

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

    def forward(self, x, context=None, context_mask=None):
        """
        Args:
            x: Input tensor of shape (B, N, D) where N is the query sequence length
            context: Context tensor of shape (B, L, D) where L is the context sequence length
            context_mask: Boolean mask of shape (B, L) where True means the value is masked

        Returns:
            Tensor of shape (B, N, D)

        Where:
            B: batch size
            N: sequence length of query
            L: sequence length of context
            D: dimension of input features
        """
        h = self.heads
        q = self.to_q(x)

        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        # where d = dim_head
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        # sim shape: (B*h, N, L)
        sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

        if context_mask is not None:
            # Expand mask for the heads dimension
            # context_mask shape: (B, L) -> (B, 1, L) -> (B*h, 1, L)
            mask = context_mask.unsqueeze(1).expand(-1, 1, -1)
            mask = mask.repeat_interleave(h, dim=0)

            # Expand mask for the query dimension
            # mask shape: (B*h, 1, L) -> (B*h, N, L)
            mask = mask.expand(-1, x.size(1), -1)

            # Create a mask of -inf where context_mask is True
            mask_value = -torch.finfo(sim.dtype).max
            sim = sim.masked_fill(mask, mask_value)

        # attn shape: (B*h, N, L)
        attn = sim.softmax(dim=-1)

        # out shape: (B*h, N, d) -> (B, N, h*d)
        out = torch.einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)

        return self.to_out(out)


class Attention(nn.Module):
    """
    Attention module with optional drop path.
    """

    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        drop_path_rate=0.0,
        use_geglu=True,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        if mask is not None:
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)
        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.drop_path(self.to_out(out))


class LatentTransformerBlock(nn.Module):
    """
    Basic latent transformer block with self-attention and feedforward.
    """

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
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
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )

        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )

        self.ls3 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )

    def forward(self, x, context=None, context_mask=None):
        x = self.ls1(self.attn1(self.norm1(x))) + x
        x = self.ls2(self.attn2(self.norm2(x), context=context, context_mask=None)) + x
        x = self.ls3(self.ff(self.norm3(x))) + x
        return x


class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        dim_head,
        drop_path_rate=0.0,
        use_geglu=True,
    ):
        super().__init__()
        self.attn = PreNorm(
            dim,
            Attention(
                dim,
                heads=heads,
                dim_head=dim_head,
                drop_path_rate=drop_path_rate,
                use_geglu=use_geglu,
            ),
        )
        self.ff = PreNorm(dim, FeedForward(dim, drop_path_rate=drop_path_rate))

    def forward(self, x, context=None, mask=None):
        x = self.attn(x, context=context, mask=mask) + x
        x = self.ff(x) + x
        return x


class StackedAttentionBlocks(nn.Module):
    def __init__(
        self, dim, depth, heads, dim_head, weight_tie_layers=False, use_geglu=True
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                AttentionBlock(dim, heads, dim_head, use_geglu=use_geglu)
            )

        if weight_tie_layers:
            self.layers = nn.ModuleList([self.layers[0]] * depth)

    def forward(self, x, context=None, mask=None):
        for layer in self.layers:
            x = layer(x, context=context, mask=mask)
        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = (
            nn.LayerNorm(context_dim) if context_dim is not None else None
        )

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if self.norm_context is not None:
            context = kwargs.get("context", x)
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)
