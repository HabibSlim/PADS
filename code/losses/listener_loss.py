"""
Neural network listener loss on shape pairs.
"""
import torch
import torch.nn.functional as F


class ListenerLoss:
    def __init__(self):
        pass

    def __call__(self, net, x_a, x_b, embed_ab, labels):
        """
        Call the loss function.

        net: the latent diffusion model
        x_a, x_b: the set of AE latents of dimensions B x M x C, where M = 512 and C = 8
        embed_ab: the text embedding of dimensions B x M, where M = 768
        labels: the binary labels of dimensions B x 1
        """
        # Reshape from (B, D, K) to (B, M)
        x_a = x_a.flatten(1)
        x_b = x_b.flatten(1)
        embed_ab = embed_ab.flatten(1)

        # Concatenate the latent vector with the embedding
        pred = net(x_a, x_b, embed_ab)

        # Compute cross-entropy loss
        loss = F.binary_cross_entropy_with_logits(pred, labels)

        # Compute accuracy
        pred = torch.sigmoid(pred)
        # acc = (pred > 0.5).eq(labels).float().mean()

        return loss.mean(), pred
