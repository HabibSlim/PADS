"""
Generic PyTorch modules.
"""

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath
from torch import einsum, nn
from util.misc import default


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, drop_path_rate=0.0, use_geglu=True):
        super().__init__()
        if use_geglu:
            self.net = nn.Sequential(
                nn.Linear(dim, dim * mult * 2), GEGLU(), nn.Linear(dim * mult, dim)
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(dim, dim * mult), nn.ReLU(), nn.Linear(dim * mult, dim)
            )

        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def forward(self, x):
        return self.drop_path(self.net(x))


class Attention(nn.Module):
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

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.drop_path(self.to_out(out))


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

    def forward(self, x):
        x = self.attn(x) + x
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

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class PointEmbed(nn.Module):
    def __init__(self, hidden_dim=48, dim=128):
        super().__init__()

        assert hidden_dim % 6 == 0

        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack(
            [
                torch.cat(
                    [
                        e,
                        torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6),
                    ]
                ),
                torch.cat(
                    [
                        torch.zeros(self.embedding_dim // 6),
                        e,
                        torch.zeros(self.embedding_dim // 6),
                    ]
                ),
                torch.cat(
                    [
                        torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6),
                        e,
                    ]
                ),
            ]
        )
        self.register_buffer("basis", e)  # 3 x 16

        self.mlp = nn.Linear(self.embedding_dim + 3, dim)

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum("bnd,de->bne", input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings

    def forward(self, input):
        # input: B x N x 3
        embed = self.mlp(
            torch.cat([self.embed(input, self.basis), input], dim=2)
        )  # B x N x C
        return embed


class CartesianPosEmbed(nn.Module):
    def __init__(self, n_latents, latent_dim):
        super().__init__()
        self.projection = nn.Conv2d(4, n_latents, 1)
        self.register_buffer("pe", self.build_grid(latent_dim).unsqueeze(0))

    def build_grid(self, side_length):
        coords = torch.linspace(0.0, 1.0, side_length + 1)
        coords = 0.5 * (coords[:-1] + coords[1:])
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing="xy")
        return torch.stack((grid_x, grid_y, 1 - grid_x, 1 - grid_y), dim=0)

    def forward(self, inputs):
        return inputs + self.projection(self.pe)


class DiagonalGaussianDistribution(object):
    def __init__(self, mean, logvar, deterministic=False, no_reduction=False):
        self.mean = mean
        self.logvar = logvar
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(
                device=self.mean.device
            )
        self.no_reduction = no_reduction

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(
            device=self.mean.device
        )
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                kl_vec = 0.5 * (torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar)
                if self.no_reduction:
                    return kl_vec
                else:
                    return torch.mean(
                        kl_vec,
                        dim=[1, 2],
                    )
            else:
                kl_vec = 0.5 * (
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                )
                if self.no_reduction:
                    return kl_vec
                else:
                    return torch.mean(
                        kl_vec,
                        dim=[1, 2, 3],
                    )

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self):
        return self.mean
