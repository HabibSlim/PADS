"""
Node to node loss function for the latent diffusion model,
with a prompt.
"""
import torch


class Node2NodeLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=1):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, x_a, x_b, embed_ab):
        """
        Call the loss function.

        net: the latent diffusion model
        x_a, x_b: the sets for original and edited nodes, 2 x [M:512 x C:512]
        sigma: the noise level of the diffusion process, [M x 1 x 1]
        embed_ab: the prompt embedding from a to b, [B x 1 x 512]
        """
        rnd_normal = torch.randn([x_b.shape[0], 1, 1], device=x_a.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

        n = torch.randn_like(x_b) * sigma
        noised_x_b = x_b + n

        D_x_b = net(x_a=x_a, x_b=noised_x_b, embeds_ab=embed_ab, sigma=sigma)
        loss = weight * ((D_x_b - x_b) ** 2)

        # Clip the loss to avoid NaN
        loss = torch.clamp(loss, min=-1e3, max=1e3)
        return loss.mean()
