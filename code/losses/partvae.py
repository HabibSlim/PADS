"""
Defining loss functions used for the PartVAE model.
"""

import torch
from torch import nn


class KLRecLoss:
    """
    Kullback-Leibler divergence loss for the PartVAE model.
    """

    def __init__(self):
        pass

    def __call__(self, kl):
        """
        Call the loss function.
        """
        loss_kl = torch.sum(kl) / kl.shape[0]
        return loss_kl.mean()


class ScaleInvariantLoss:
    """
    Scale-invariant loss for the PartVAE model.
    """

    def __init__(self, l2_loss):
        self.l2_loss = l2_loss

    def __call__(self, part_latents_a, part_latents_b, bb_a, bb_b, mask_a, mask_b):
        """
        Call the loss function.
        """
        permuts = self.find_permutation_padded(bb_a, bb_b, mask_a, mask_b)
        total_loss = 0.0
        inter = mask_a & mask_b
        B = permuts.shape[0]
        for i in range(B):
            b_to_a = permuts[i][inter[i]]
            n_parts = len(b_to_a)
            # assert (bb_b[i][b_to_a] == bb_a[i][:n_parts]).all()
            total_loss += (
                self.l2_loss(part_latents_b[i][b_to_a], part_latents_a[i][:n_parts])
                / n_parts
            )
        return total_loss / B


class PartDropLoss:
    """
    Part drop loss for the PartVAE model.
    """

    def __init__(self, l2_loss):
        self.l2_loss = l2_loss

    def find_permutation_padded(self, bb_a, bb_b, mask_a, mask_b):
        """
        Find the permutation matrix between two sets of items.
        """
        n, m, p, q = bb_a.shape
        permutation = torch.full((n, m), -1, dtype=torch.long, device=bb_a.device)

        for i in range(n):
            A_items = bb_a[i, mask_a[i]]
            B_items = bb_b[i, mask_b[i]]

            # Reshape items for comparison
            A_flat = A_items.reshape(-1, p * q)
            B_flat = B_items.reshape(-1, p * q)

            # Compute equality matrix
            equality_matrix = (A_flat.unsqueeze(1) == B_flat.unsqueeze(0)).all(dim=-1)

            # Find matches
            A_matches, B_matches = torch.where(equality_matrix)

            # Update permutation
            A_indices = torch.nonzero(mask_a[i]).squeeze()
            B_indices = torch.nonzero(mask_b[i]).squeeze()
            permutation[i, A_indices[A_matches]] = B_indices[B_matches]

        return permutation

    def __call__(self, part_latents_a, part_latents_b, bb_a, bb_b, mask_a, mask_b):
        """
        Call the loss function.
        """
        permuts = self.find_permutation_padded(bb_a, bb_b, mask_a, mask_b)
        total_loss = 0.0
        inter = mask_a & mask_b
        B = permuts.shape[0]
        for i in range(B):
            b_to_a = permuts[i][inter[i]]
            n_parts = len(b_to_a)
            # assert (bb_b[i][b_to_a] == bb_a[i][:n_parts]).all()
            total_loss += (
                self.l2_loss(part_latents_b[i][b_to_a], part_latents_a[i][:n_parts])
                / n_parts
            )
        return total_loss / B
