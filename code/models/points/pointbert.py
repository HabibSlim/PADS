"""
PointNet++ model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

import models.points.pointbert_utils as p2u
from knn_cuda import KNN


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, data):
        """
        input: B N 3
        ---------------------------
        output: B G M 3
        center : B G 3
        """
        batch_size, num_points, C = data.shape
        if C > 3:
            # data = xyz
            xyz = data[:, :, :3].contiguous()
            rgb = data[:, :, 3:].contiguous()
        else:
            xyz = data.contiguous()
        # fps the centers out
        center = p2u.fps(xyz, self.num_group)  # B G 3
        print(center.shape)
        assert center.size(1) == self.num_group
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center)  # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = (
            torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        )
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood_xyz = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood_xyz = neighborhood_xyz.view(
            batch_size, self.num_group, self.group_size, 3
        ).contiguous()
        if C > 3:
            neighborhood_rgb = rgb.view(batch_size * num_points, -1)[idx, :]
            neighborhood_rgb = neighborhood_rgb.view(
                batch_size, self.num_group, self.group_size, 3
            ).contiguous()

        # normalize xyz
        neighborhood_xyz = neighborhood_xyz - center.unsqueeze(2)
        if C > 3:
            features = torch.cat((neighborhood_xyz, neighborhood_rgb), dim=-1)
            return features, center
        else:
            return neighborhood_xyz, center


class Encoder(nn.Module):
    def __init__(self, encoder_channel, point_input_dims=3):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.point_input_dims = point_input_dims
        self.first_conv = nn.Sequential(
            nn.Conv1d(self.point_input_dims, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1),
        )

    def forward(self, point_groups):
        """
        point_groups : B G N 3
        -----------------
        feature_global : B G C
        """
        bs, g, n, c = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, c)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat(
            [feature_global.expand(-1, -1, n), feature], dim=1
        )  # BG 512 n
        feature = self.second_conv(feature)  # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder without hierarchical structure.
    """

    def __init__(
        self,
        embed_dim=768,
        depth=4,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=(
                        drop_path_rate[i]
                        if isinstance(drop_path_rate, list)
                        else drop_path_rate
                    ),
                )
                for i in range(depth)
            ]
        )

    def forward(self, x, pos):
        feature_list = []
        fetch_idx = [3, 7, 11]
        for i, block in enumerate(self.blocks):
            x = block(x + pos)
            if i in fetch_idx:
                feature_list.append(x)
        return feature_list


class DGCNN_Propagation(nn.Module):
    def __init__(self, k=16):
        super().__init__()
        """
        K has to be 16
        """
        self.k = k
        self.knn = KNN(k=k, transpose_mode=False)

        self.layer1 = nn.Sequential(
            nn.Conv2d(768, 512, kernel_size=1, bias=False),
            nn.GroupNorm(4, 512),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(1024, 384, kernel_size=1, bias=False),
            nn.GroupNorm(4, 384),
            nn.LeakyReLU(negative_slope=0.2),
        )

    @staticmethod
    def fps_downsample(coor, x, num_group):
        xyz = coor.transpose(1, 2).contiguous()  # b, n, 3
        fps_idx = p2u.furthest_point_sample(xyz, num_group)

        combined_x = torch.cat([coor, x], dim=1)

        new_combined_x = p2u.gather_operation(combined_x, fps_idx)

        new_coor = new_combined_x[:, :3]
        new_x = new_combined_x[:, 3:]

        return new_coor, new_x

    def get_graph_feature(self, coor_q, x_q, coor_k, x_k):
        k = self.k
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
            _, idx = self.knn(coor_k, coor_q)  # bs k np
            assert idx.shape[1] == k
            idx_base = (
                torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1)
                * num_points_k
            )
            idx = idx + idx_base
            idx = idx.view(-1)
        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = (
            feature.view(batch_size, k, num_points_q, num_dims)
            .permute(0, 3, 2, 1)
            .contiguous()
        )
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature

    def forward(self, coor, f, coor_q, f_q):
        """coor, f : B 3 G ; B C G
        coor_q, f_q : B 3 N; B 3 N
        """
        # dgcnn upsample
        f_q = self.get_graph_feature(coor_q, f_q, coor, f)
        f_q = self.layer1(f_q)
        f_q = f_q.max(dim=-1, keepdim=False)[0]

        f_q = self.get_graph_feature(coor_q, f_q, coor_q, f_q)
        f_q = self.layer2(f_q)
        f_q = f_q.max(dim=-1, keepdim=False)[0]

        return f_q


class PointBERT(nn.Module):
    def __init__(
        self,
        trans_dim,
        depth,
        drop_path_rate,
        num_heads,
        group_size,
        num_group,
        encoder_dims,
        point_input_dims,
        num_classes,
        num_parts,
        shape_prior=False,
    ):
        super().__init__()
        self.trans_dim = trans_dim
        self.depth = depth
        self.drop_path_rate = drop_path_rate

        self.num_heads = num_heads
        self.shape_prior = shape_prior
        self.group_size = group_size
        self.num_group = num_group

        # Data parameters
        self.num_classes = num_classes
        self.num_parts = num_parts

        # Grouper
        self.group_divider = Group(num_group=num_group, group_size=group_size)

        # Encoder
        self.encoder = Encoder(
            encoder_channel=encoder_dims,
            point_input_dims=point_input_dims,
        )
        self.reduce_dim = nn.Linear(encoder_dims, trans_dim)

        # Tokens and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, trans_dim))
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, trans_dim)
        )

        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = TransformerEncoder(
            embed_dim=trans_dim,
            depth=depth,
            drop_path_rate=dpr,
            num_heads=num_heads,
        )

        self.norm = nn.LayerNorm(trans_dim)

        # Feature propagation layers
        self.propagation_2 = p2u.PointNetFeaturePropagation(
            in_channel=trans_dim + 3, mlp=[trans_dim * 4, trans_dim]
        )
        self.propagation_1 = p2u.PointNetFeaturePropagation(
            in_channel=trans_dim + 3, mlp=[trans_dim * 4, trans_dim]
        )
        self.propagation_0 = p2u.PointNetFeaturePropagation(
            in_channel=trans_dim + 3 + (num_classes if shape_prior else 0),
            mlp=[trans_dim * 4, trans_dim],
        )

        # DGCNN propagation layers
        self.dgcnn_pro_1 = DGCNN_Propagation(k=4)
        self.dgcnn_pro_2 = DGCNN_Propagation(k=4)

        # Final convolution layers
        self.conv1 = nn.Conv1d(trans_dim, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_parts, 1)

    def forward(self, pts, cls_label=None):
        """
        pts       : B N 3
        cls_label : B N
        """
        B, N, C = pts.shape
        neighborhood, center = self.group_divider(pts)

        group_input_tokens = self.encoder(neighborhood)  # B G N
        group_input_tokens = self.reduce_dim(group_input_tokens)

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = self.pos_embed(center)
        pos = torch.cat((cls_pos, pos), dim=1)

        feature_list = self.blocks(x, pos)

        feature_list = [
            self.norm(x)[:, 1:].transpose(-1, -2).contiguous() for x in feature_list
        ]

        xyz = pts[:, :, :3].contiguous()
        if self.shape_prior:
            assert cls_label is not None, "Shape prior requires class labels"
            cls_label = F.one_hot(cls_label, num_classes=self.num_classes).float()
            cls_label_one_hot = cls_label.view(B, self.num_classes, 1).repeat(1, 1, N)
            center_level_0 = xyz.transpose(-1, -2).contiguous()
            f_level_0 = torch.cat([cls_label_one_hot, center_level_0], 1)
        else:
            center_level_0 = xyz.transpose(-1, -2).contiguous()
            f_level_0 = center_level_0

        center_level_1 = p2u.fps(xyz.contiguous(), 512).transpose(-1, -2).contiguous()
        f_level_1 = center_level_1
        center_level_2 = p2u.fps(xyz.contiguous(), 256).transpose(-1, -2).contiguous()
        f_level_2 = center_level_2
        center_level_3 = center.transpose(-1, -2).contiguous()

        f_level_3 = feature_list[2]
        f_level_2 = self.propagation_2(
            center_level_2, center_level_3, f_level_2, feature_list[1]
        )
        f_level_1 = self.propagation_1(
            center_level_1, center_level_3, f_level_1, feature_list[0]
        )

        # Bottom-up propagation
        f_level_2 = self.dgcnn_pro_2(
            center_level_3, f_level_3, center_level_2, f_level_2
        )
        f_level_1 = self.dgcnn_pro_1(
            center_level_2, f_level_2, center_level_1, f_level_1
        )
        f_level_0 = self.propagation_0(
            center_level_0, center_level_1, f_level_0, f_level_1
        )

        # FC layers
        feat_bottom = F.relu(self.bn1(self.conv1(f_level_0)))
        x = self.drop1(feat_bottom)
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)

        top_fts = feat_bottom.transpose(-1, -2).contiguous()
        return x, top_fts


def pointbert_g512_d12(num_classes, num_parts, shape_prior=False):
    config = {
        "num_classes": num_classes,
        "num_parts": num_parts,
        "shape_prior": True,
        "trans_dim": 384,
        "depth": 12,
        "drop_path_rate": 0.1,
        "num_heads": 6,
        "group_size": 32,
        "num_group": 512,
        "encoder_dims": 256,
        "point_input_dims": 3,
    }
    model = PointBERT(**config).cuda()
    model.apply(p2u.inplace_relu)

    return model
