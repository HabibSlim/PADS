"""
Defining loss functions used for the PartVAE model.
"""

import torch


class KLRecLoss:
    """
    Kullback-Leibler divergence loss for the PartVAE model.
    """

    def __init__(self, kl_weight=1.0):
        self.kl_weight = kl_weight

    def __call__(self, kl):
        """
        Call the loss function.
        """
        loss_kl = torch.sum(kl) / kl.shape[0]
        return self.kl_weight * loss_kl.mean()
