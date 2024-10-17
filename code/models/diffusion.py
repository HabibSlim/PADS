"""
Transformer-based diffusion model for text-conditioned shape editing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import models.sampling as smp


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

    def forward(self, x, context=None):
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


class AdaLayerNorm(nn.Module):
    def __init__(self, n_embd):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd * 2)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)

    def forward(self, x, timestep):
        emb = self.linear(timestep)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.layernorm(x) * (1 + scale) + shift
        return x


class BasicTransformerBlock(nn.Module):
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

    def forward(self, x, t, context=None):
        x = self.ls1(self.attn1(self.norm1(x, t))) + x
        x = self.ls2(self.attn2(self.norm2(x, t), context=context)) + x
        x = self.ls3(self.ff(self.norm3(x, t))) + x
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
        self.map_noise = PositionalEmbedding(t_channels)

        self.map_layer0 = nn.Linear(in_features=t_channels, out_features=inner_dim)
        self.map_layer1 = nn.Linear(in_features=inner_dim, out_features=inner_dim)

    def forward(self, x, t, cond=None):
        t_emb = self.map_noise(t)[:, None]
        t_emb = F.silu(self.map_layer0(t_emb))
        t_emb = F.silu(self.map_layer1(t_emb))
        x = self.proj_in(x)

        for block in self.transformer_blocks:
            x = block(x, t_emb, context=cond)

        x = self.norm(x)
        x = self.proj_out(x)
        return x


class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [
            torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds
        ]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack(
            [torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators]
        )

    def randn_like(self, input):
        return self.randn(
            input.shape, dtype=input.dtype, layout=input.layout, device=input.device
        )

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack(
            [
                torch.randint(*args, size=size[1:], generator=gen, **kwargs)
                for gen in self.generators
            ]
        )


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

        self.model = LatentArrayTransformer(
            in_channels=channels,
            t_channels=256,
            n_heads=n_heads,
            d_head=d_head,
            depth=depth,
            out_channels=self.out_channels,
        )
        self.dtype = torch.float16 if use_fp16 else torch.float32

    def edm_forward(self, x, sigma, cond):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4
        c_noise = c_noise.to(self.dtype)

        F_x = self.model((c_in * x).to(self.dtype), c_noise.flatten(), cond)
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
        cond, part_queries = self.pqe(x, part_bbs, part_labels, batch_mask)

        return self.edm_forward(x=x, sigma=sigma, cond=cond), part_queries


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
    layer_depth, n_parts=24, part_latents_dim=512, single_learnable_query=False
):
    from models.part_queries import PQM, PartQueriesEncoder

    pqm = PQM(
        dim=512,
        latent_dim=part_latents_dim,
        heads=8,
        dim_head=64,
        depth=layer_depth,
        single_learnable_query=single_learnable_query,
    )
    pqe = PartQueriesEncoder(
        pqm=pqm,
        dim=part_latents_dim,
        input_length=n_parts,
        output_length=8,
    )
    model = EDMPartQueries(
        part_queries_encoder=pqe, n_latents=512, channels=8, depth=24
    )
    return model


def kl_d512_m512_l8_d24_pq_shallow(layer_depth=None, n_parts=24, part_latents_dim=512):
    from models.part_queries import PQMShallow, PartQueriesEncoder

    pqm = PQMShallow(
        dim=512,
        latent_dim=part_latents_dim,
        heads=8,
        dim_head=64,
        use_attention_masking=False,
    )
    pqe = PartQueriesEncoder(
        pqm=pqm,
        dim=part_latents_dim,
        input_length=n_parts,
        output_length=8,
    )
    model = EDMPartQueries(
        part_queries_encoder=pqe, n_latents=512, channels=8, depth=24
    )
    return model
