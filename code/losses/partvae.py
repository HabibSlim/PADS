"""
Defining loss functions used for the PartVAE model.
"""

import torch
from torch.nn import functional as F
from torch_linear_assignment import batch_linear_assignment


class KLRecLoss:
    """
    Kullback-Leibler divergence loss for the PartVAE model.
    """

    def __init__(self):
        pass

    def __call__(self, kl, mask=None):
        """
        Call the loss function.
        """
        if mask is not None:
            kl = kl[mask]
        loss_kl = torch.sum(kl) / kl.shape[0]
        return loss_kl.mean()


class RecLoss:
    """
    Set reconstruction loss for the global latents.
    Each x and x_rec are of shape B x 512 x 8.
    First match the vectors in sets x and x_rec
    using the linear assignment algorithm (each latent vector is of size 512).
    Then, calculate the MSE loss between the matched vectors.
    Return the mean of the losses.
    """

    def __init__(self):
        pass

    def __call__(self, x, x_rec, transpose=False):
        """
        Call the loss function.
        """
        if transpose:
            x = x.transpose(1, 2)
            x_rec = x_rec.transpose(1, 2)

        B, N, D = (
            x.shape
        )  # B: batch size, N: number of vectors (8), D: latent dimension (512)

        # Compute the cost matrix using cdist
        cost_matrix = torch.cdist(x, x_rec, p=2)

        # Compute the linear assignment
        assignment = batch_linear_assignment(cost_matrix)

        # Compute the loss
        x_rec_matched = x_rec[torch.arange(B).unsqueeze(1), assignment]
        return F.mse_loss(x, x_rec_matched)


class ScaleInvariantLoss:
    """
    Scale-invariant loss for the PartVAE model.
    """

    def __init__(self):
        pass

    def __call__(self, part_latents_a, part_latents_b, mask):
        """
        Call the loss function.
        """
        # Apply mask to both latents
        masked_latents_a = part_latents_a[mask]
        masked_latents_b = part_latents_b[mask]

        # Calculate L2 loss
        return F.mse_loss(masked_latents_a, masked_latents_b)


class PartDropLoss:
    """
    Part drop loss for the PartVAE model.
    """

    def __init__(self):
        pass

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

            # Deal with the case where A_indices is a single element
            if A_indices.dim() == 0:
                A_indices = A_indices.unsqueeze(0)
            if B_indices.dim() == 0:
                B_indices = B_indices.unsqueeze(0)
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
            total_loss += F.mse_loss(
                part_latents_b[i][b_to_a], part_latents_a[i][:n_parts]
            )
        return total_loss / B
