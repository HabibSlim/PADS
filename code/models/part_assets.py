"""
Defining a part-neural asset model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.points.pointbert_utils import fps


class WeightedAggregation(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        self.weight_layer = nn.Linear(feature_dim, 1)

    def forward(self, x):
        # Compute weights for each feature vector
        weights = self.weight_layer(x)  # B, N_p, 512, 1

        # Apply softmax to get a probability distribution
        weights = F.softmax(weights, dim=-1)  # B, N_p, 512, 1

        # Apply weights to input vectors and sum
        weighted_sum = torch.mean(weights * x, dim=2)  # B, N_p, 128

        return weighted_sum


class BoundingBoxTokenizer(nn.Module):
    def __init__(
        self, bb_input_dim=12, mlp_hidden_dim=64, mlp_output_dim=32, mlp_depth=3
    ):
        super().__init__()
        self.mlp = self._build_mlp(
            bb_input_dim, mlp_hidden_dim, mlp_output_dim, mlp_depth
        )
        self.output_dim = mlp_output_dim

    def _build_mlp(self, input_dim, hidden_dim, output_dim, depth):
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(depth - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        return nn.Sequential(*layers)

    def forward(self, part_bbs):
        B, P, _, _ = part_bbs.shape  # B, P, 4, 3

        # Flatten the bounding boxes
        flattened_bbs = part_bbs.view(B, P, -1)  # B, P, 12

        # Process through MLP
        pose_tokens = self.mlp(flattened_bbs)  # B, P, mlp_output_dim

        return pose_tokens


class PartTokenizer(nn.Module):
    def __init__(
        self,
        pc_encoder,
        bb_input_dim=12,
        bb_hidden_dim=64,
        bb_output_dim=32,
        bb_mlp_depth=3,
        visual_feature_dim=128,
        out_dim=512,
    ):
        super().__init__()
        self.pc_encoder = pc_encoder
        self.pc_encoder.train(True)

        self.bb_tokenizer = BoundingBoxTokenizer(
            bb_input_dim=bb_input_dim,
            mlp_hidden_dim=bb_hidden_dim,
            mlp_output_dim=bb_output_dim,
            mlp_depth=bb_mlp_depth,
        )
        self.visual_aggregator = WeightedAggregation(feature_dim=visual_feature_dim)
        self.visual_feature_dim = visual_feature_dim
        self.output_dim = bb_output_dim + visual_feature_dim
        self.out_proj = nn.Linear(self.output_dim, out_dim)

    def forward(
        self,
        part_bbs,
        part_points,
        batch_mask,  # 1 if part is invalid, 0 otherwise
        shape_cls=None,
        num_samples=512,
        deterministic=False,
    ):
        flipped_mask = ~batch_mask
        B, P = flipped_mask.shape

        # Apply mask to bounding boxes and get tokens
        bb_tokens = self.bb_tokenizer(part_bbs)  # B, P, bb_output_dim
        bb_tokens = bb_tokens * flipped_mask.unsqueeze(-1)  # Zero out invalid parts

        # Process point clouds batch-wise
        N_s = num_samples
        part_fts = torch.zeros(
            B, P, N_s, self.visual_feature_dim, device=part_points.device
        )
        for b in range(B):
            valid_count = flipped_mask[b].sum().item()
            # Process valid parts for this shape
            points = part_points[b : b + 1, :valid_count]  # 1, valid_count, N, C
            resampled = self.subsample_parts(
                points, num_samples=N_s, deterministic=deterministic
            )  # 1, valid_count, N_s, C
            cur_cls = None if shape_cls is None else shape_cls[b : b + 1]
            top_fts = self.pc_encoder(resampled, cls_label=cur_cls)
            part_fts[b, :valid_count] = top_fts.view(valid_count, N_s, -1)

        # Apply weighted aggregation to get visual tokens
        visual_tokens = self.visual_aggregator(part_fts)  # B, P, visual_feature_dim
        visual_tokens = visual_tokens * flipped_mask.unsqueeze(
            -1
        )  # Zero out invalid parts

        # Concatenate bounding box tokens and visual tokens
        combined_tokens = torch.cat(
            [bb_tokens, visual_tokens], dim=-1
        )  # B, P, bb_output_dim + visual_feature_dim

        return self.out_proj(combined_tokens), (bb_tokens, visual_tokens)

    def load_encoder_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path)
        self.pc_encoder.load_state_dict(ckpt)

    @staticmethod
    @torch.no_grad()
    def subsample_parts(part_points, num_samples=512, deterministic=False):
        N_s = num_samples
        B, P, N, C = part_points.shape
        resampled = torch.zeros((B, P, N_s, C)).to(part_points.device)
        if deterministic:
            for i, p in enumerate(part_points):
                resampled[i] = p[:, :N_s]
        else:
            for i, p in enumerate(part_points):
                resampled[i] = fps(p, N_s)
        return resampled.view(B, P * N_s, C)
