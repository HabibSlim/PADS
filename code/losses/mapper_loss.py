"""
Simple latent mapping loss.
"""


class MapperLoss:
    def __init__(self):
        pass

    def __call__(self, net, x_a, x_b, embed_ab):
        """
        Call the loss function.

        net: the latent diffusion model
        x_a: the set of AE latents of dimensions M x C, where M = 512 and C = 512
        sigma: the noise level of the diffusion process, of dimension M x 1 x 1
        """
        # Reshape from (B, D, K) to (B, M)
        x_a = x_a.flatten(1)
        x_b = x_b.flatten(1)
        embed_ab = embed_ab.flatten(1)

        # Concatenate the latent vector with the embedding
        edit_vec = net(x_a, embed_ab)

        # Add the edit vector to the latent vector
        x_bp = x_a + edit_vec

        # Compute the L2 loss
        loss = ((x_b - x_bp) ** 2).mean(dim=1)

        return loss.mean()


class MapperLossDirect:
    def __init__(self):
        pass

    def __call__(self, net, x_a, x_b, embed_ab):
        """
        Call the loss function.

        net: the latent diffusion model
        x_a: the set of AE latents of dimensions M x C, where M = 512 and C = 512
        sigma: the noise level of the diffusion process, of dimension M x 1 x 1
        """
        # Reshape from (B, D, K) to (B, M)
        x_a = x_a.flatten(1)
        x_b = x_b.flatten(1)
        embed_ab = embed_ab.flatten(1)

        # Directly map x_a to x_b
        x_bp = net(x_a, embed_ab)

        # Compute the L2 loss
        loss = ((x_b - x_bp) ** 2).mean(dim=1)

        return loss.mean()
