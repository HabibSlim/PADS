"""
Diffusion loss function for the latent diffusion model conditioned on part queries.
"""

import torch


class EDMLossPQDM:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=1):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, x, part_bbs, part_labels, batch_mask):
        """
        Call the loss function.

        net: the latent diffusion model
        x_a: the set of AE latents of dimensions M x C, where M = 512 and C = 512
        sigma: the noise level of the diffusion process, of dimension M x 1 x 1
        """
        rnd_normal = torch.randn([x.shape[0], 1, 1], device=x.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        n = torch.randn_like(x) * sigma

        samples = (x + n, part_bbs, part_labels, batch_mask)
        D_yn, part_queries = net(samples=samples, sigma=sigma)
        loss = weight * ((D_yn - x) ** 2)
        return loss.mean(), part_queries
