"""
Diffusion loss function for the latent diffusion model.
"""
import torch


class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=1):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, x_a, embed=None):
        """
        Call the loss function.

        net: the latent diffusion model
        x_a: the set of AE latents of dimensions M x C, where M = 512 and C = 512
        sigma: the noise level of the diffusion process, of dimension M x 1 x 1
        """
        rnd_normal = torch.randn([x_a.shape[0], 1, 1], device=x_a.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

        n = torch.randn_like(x_a) * sigma

        D_yn = net(x_a + n, sigma, embed)
        loss = weight * ((D_yn - x_a) ** 2)
        return loss.mean()
