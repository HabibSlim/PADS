"""
IMNet implementation.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mcubes
from eval.sampling_util import sample_pc_random


class generator(nn.Module):
    def __init__(self, z_dim, point_dim, gf_dim):
        super(generator, self).__init__()
        self.z_dim = z_dim
        self.point_dim = point_dim
        self.gf_dim = gf_dim
        self.linear_1 = nn.Linear(
            self.z_dim + self.point_dim, self.gf_dim * 8, bias=True
        )
        self.linear_2 = nn.Linear(self.gf_dim * 8, self.gf_dim * 8, bias=True)
        self.linear_3 = nn.Linear(self.gf_dim * 8, self.gf_dim * 8, bias=True)
        self.linear_4 = nn.Linear(self.gf_dim * 8, self.gf_dim * 4, bias=True)
        self.linear_5 = nn.Linear(self.gf_dim * 4, self.gf_dim * 2, bias=True)
        self.linear_6 = nn.Linear(self.gf_dim * 2, self.gf_dim * 1, bias=True)
        self.linear_7 = nn.Linear(self.gf_dim * 1, 1, bias=True)
        nn.init.normal_(self.linear_1.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_1.bias, 0)
        nn.init.normal_(self.linear_2.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_2.bias, 0)
        nn.init.normal_(self.linear_3.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_3.bias, 0)
        nn.init.normal_(self.linear_4.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_4.bias, 0)
        nn.init.normal_(self.linear_5.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_5.bias, 0)
        nn.init.normal_(self.linear_6.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_6.bias, 0)
        nn.init.normal_(self.linear_7.weight, mean=1e-5, std=0.02)
        nn.init.constant_(self.linear_7.bias, 0)

    def forward(self, points, z):
        zs = z.view(-1, 1, self.z_dim).repeat(1, points.size()[1], 1)
        pointz = torch.cat([points, zs], 2)

        l1 = self.linear_1(pointz)
        l1 = F.leaky_relu(l1, negative_slope=0.02, inplace=True)

        l2 = self.linear_2(l1)
        l2 = F.leaky_relu(l2, negative_slope=0.02, inplace=True)

        l3 = self.linear_3(l2)
        l3 = F.leaky_relu(l3, negative_slope=0.02, inplace=True)

        l4 = self.linear_4(l3)
        l4 = F.leaky_relu(l4, negative_slope=0.02, inplace=True)

        l5 = self.linear_5(l4)
        l5 = F.leaky_relu(l5, negative_slope=0.02, inplace=True)

        l6 = self.linear_6(l5)
        l6 = F.leaky_relu(l6, negative_slope=0.02, inplace=True)

        l7 = self.linear_7(l6)

        l7 = torch.max(torch.min(l7, l7 * 0.01 + 0.99), l7 * 0.01)

        return l7


class encoder(nn.Module):
    def __init__(self, ef_dim, z_dim):
        super(encoder, self).__init__()
        self.ef_dim = ef_dim
        self.z_dim = z_dim
        self.conv_1 = nn.Conv3d(1, self.ef_dim, 4, stride=2, padding=1, bias=False)
        self.in_1 = nn.InstanceNorm3d(self.ef_dim)
        self.conv_2 = nn.Conv3d(
            self.ef_dim, self.ef_dim * 2, 4, stride=2, padding=1, bias=False
        )
        self.in_2 = nn.InstanceNorm3d(self.ef_dim * 2)
        self.conv_3 = nn.Conv3d(
            self.ef_dim * 2, self.ef_dim * 4, 4, stride=2, padding=1, bias=False
        )
        self.in_3 = nn.InstanceNorm3d(self.ef_dim * 4)
        self.conv_4 = nn.Conv3d(
            self.ef_dim * 4, self.ef_dim * 8, 4, stride=2, padding=1, bias=False
        )
        self.in_4 = nn.InstanceNorm3d(self.ef_dim * 8)
        self.conv_5 = nn.Conv3d(
            self.ef_dim * 8, self.z_dim, 4, stride=1, padding=0, bias=True
        )
        nn.init.xavier_uniform_(self.conv_1.weight)
        nn.init.xavier_uniform_(self.conv_2.weight)
        nn.init.xavier_uniform_(self.conv_3.weight)
        nn.init.xavier_uniform_(self.conv_4.weight)
        nn.init.xavier_uniform_(self.conv_5.weight)
        nn.init.constant_(self.conv_5.bias, 0)

    def forward(self, inputs):
        d_1 = self.in_1(self.conv_1(inputs))
        d_1 = F.leaky_relu(d_1, negative_slope=0.02, inplace=True)

        d_2 = self.in_2(self.conv_2(d_1))
        d_2 = F.leaky_relu(d_2, negative_slope=0.02, inplace=True)

        d_3 = self.in_3(self.conv_3(d_2))
        d_3 = F.leaky_relu(d_3, negative_slope=0.02, inplace=True)

        d_4 = self.in_4(self.conv_4(d_3))
        d_4 = F.leaky_relu(d_4, negative_slope=0.02, inplace=True)

        d_5 = self.conv_5(d_4)
        d_5 = d_5.view(-1, self.z_dim)
        d_5 = torch.sigmoid(d_5)

        return d_5


class IMNet(nn.Module):
    """
    IMNet implementation.
    """

    def __init__(self, ef_dim, gf_dim, z_dim, point_dim):
        super(IMNet, self).__init__()
        self.ef_dim = ef_dim
        self.gf_dim = gf_dim
        self.z_dim = z_dim
        self.point_dim = point_dim
        self.encoder = encoder(self.ef_dim, self.z_dim)
        self.generator = generator(self.z_dim, self.point_dim, self.gf_dim)

    def forward(self, inputs, z_vector=None, point_coord=None):
        if inputs is not None:
            z_vector = self.encoder(inputs)
        if z_vector is not None and point_coord is not None:
            net_out = self.generator(point_coord, z_vector)
        else:
            net_out = None

        return z_vector, net_out


class IMNetAE(object):
    """
    IMNet Autoencoder implementation.
    """

    def __init__(self, sample_vox_size, device):
        self.sample_vox_size = sample_vox_size
        self.point_batch_size = 16 * 16 * 16
        self.shape_batch_size = 32
        self.input_size = 64  # input voxel grid size

        self.ef_dim = 32
        self.gf_dim = 128
        self.z_dim = 256
        self.point_dim = 3

        # build model
        self.device = device
        self.IMNet = IMNet(self.ef_dim, self.gf_dim, self.z_dim, self.point_dim)
        self.IMNet.to(self.device)

        # keep everything a power of 2
        self.cell_grid_size = 4
        self.frame_grid_size = 64
        self.real_size = (
            self.cell_grid_size * self.frame_grid_size
        )  # =256, output point-value voxel grid size in testing
        self.test_size = (
            32  # related to testing batch_size, adjust according to gpu memory size
        )
        self.test_point_batch_size = (
            self.test_size * self.test_size * self.test_size
        )  # do not change

        # get coords for training
        dima = self.test_size
        dim = self.frame_grid_size
        self.aux_x = np.zeros([dima, dima, dima], np.uint8)
        self.aux_y = np.zeros([dima, dima, dima], np.uint8)
        self.aux_z = np.zeros([dima, dima, dima], np.uint8)
        multiplier = int(dim / dima)
        multiplier2 = multiplier * multiplier
        multiplier3 = multiplier * multiplier * multiplier
        for i in range(dima):
            for j in range(dima):
                for k in range(dima):
                    self.aux_x[i, j, k] = i * multiplier
                    self.aux_y[i, j, k] = j * multiplier
                    self.aux_z[i, j, k] = k * multiplier
        self.coords = np.zeros([multiplier3, dima, dima, dima, 3], np.float32)
        for i in range(multiplier):
            for j in range(multiplier):
                for k in range(multiplier):
                    self.coords[i * multiplier2 + j * multiplier + k, :, :, :, 0] = (
                        self.aux_x + i
                    )
                    self.coords[i * multiplier2 + j * multiplier + k, :, :, :, 1] = (
                        self.aux_y + j
                    )
                    self.coords[i * multiplier2 + j * multiplier + k, :, :, :, 2] = (
                        self.aux_z + k
                    )
        self.coords = (self.coords.astype(np.float32) + 0.5) / dim - 0.5
        self.coords = np.reshape(
            self.coords, [multiplier3, self.test_point_batch_size, 3]
        )
        self.coords = torch.from_numpy(self.coords)
        self.coords = self.coords.to(self.device)

        # get coords for testing
        dimc = self.cell_grid_size
        dimf = self.frame_grid_size
        self.cell_x = np.zeros([dimc, dimc, dimc], np.int32)
        self.cell_y = np.zeros([dimc, dimc, dimc], np.int32)
        self.cell_z = np.zeros([dimc, dimc, dimc], np.int32)
        self.cell_coords = np.zeros([dimf, dimf, dimf, dimc, dimc, dimc, 3], np.float32)
        self.frame_coords = np.zeros([dimf, dimf, dimf, 3], np.float32)
        self.frame_x = np.zeros([dimf, dimf, dimf], np.int32)
        self.frame_y = np.zeros([dimf, dimf, dimf], np.int32)
        self.frame_z = np.zeros([dimf, dimf, dimf], np.int32)
        for i in range(dimc):
            for j in range(dimc):
                for k in range(dimc):
                    self.cell_x[i, j, k] = i
                    self.cell_y[i, j, k] = j
                    self.cell_z[i, j, k] = k
        for i in range(dimf):
            for j in range(dimf):
                for k in range(dimf):
                    self.cell_coords[i, j, k, :, :, :, 0] = self.cell_x + i * dimc
                    self.cell_coords[i, j, k, :, :, :, 1] = self.cell_y + j * dimc
                    self.cell_coords[i, j, k, :, :, :, 2] = self.cell_z + k * dimc
                    self.frame_coords[i, j, k, 0] = i
                    self.frame_coords[i, j, k, 1] = j
                    self.frame_coords[i, j, k, 2] = k
                    self.frame_x[i, j, k] = i
                    self.frame_y[i, j, k] = j
                    self.frame_z[i, j, k] = k
        self.cell_coords = (
            self.cell_coords.astype(np.float32) + 0.5
        ) / self.real_size - 0.5
        self.cell_coords = np.reshape(
            self.cell_coords, [dimf, dimf, dimf, dimc * dimc * dimc, 3]
        )
        self.cell_x = np.reshape(self.cell_x, [dimc * dimc * dimc])
        self.cell_y = np.reshape(self.cell_y, [dimc * dimc * dimc])
        self.cell_z = np.reshape(self.cell_z, [dimc * dimc * dimc])
        self.frame_x = np.reshape(self.frame_x, [dimf * dimf * dimf])
        self.frame_y = np.reshape(self.frame_y, [dimf * dimf * dimf])
        self.frame_z = np.reshape(self.frame_z, [dimf * dimf * dimf])
        self.frame_coords = (self.frame_coords.astype(np.float32) + 0.5) / dimf - 0.5
        self.frame_coords = np.reshape(self.frame_coords, [dimf * dimf * dimf, 3])

        self.sampling_threshold = 0.5  # final marching cubes threshold

    def z2voxel(self, z):
        """
        Convert latent vector to voxel grid.
        """
        model_float = np.zeros(
            [self.real_size + 2, self.real_size + 2, self.real_size + 2], np.float32
        )
        dimc = self.cell_grid_size
        dimf = self.frame_grid_size

        frame_flag = np.zeros([dimf + 2, dimf + 2, dimf + 2], np.uint8)
        queue = []

        frame_batch_num = int(dimf**3 / self.test_point_batch_size)
        assert frame_batch_num > 0

        # Get frame grid values
        for i in range(frame_batch_num):
            point_coord = self.frame_coords[
                i * self.test_point_batch_size : (i + 1) * self.test_point_batch_size
            ]
            point_coord = np.expand_dims(point_coord, axis=0)
            point_coord = torch.from_numpy(point_coord)
            point_coord = point_coord.to(self.device)
            _, model_out_ = self.IMNet(None, z, point_coord)
            model_out = model_out_.detach().cpu().numpy()[0]
            x_coords = self.frame_x[
                i * self.test_point_batch_size : (i + 1) * self.test_point_batch_size
            ]
            y_coords = self.frame_y[
                i * self.test_point_batch_size : (i + 1) * self.test_point_batch_size
            ]
            z_coords = self.frame_z[
                i * self.test_point_batch_size : (i + 1) * self.test_point_batch_size
            ]
            frame_flag[x_coords + 1, y_coords + 1, z_coords + 1] = np.reshape(
                (model_out > self.sampling_threshold).astype(np.uint8),
                [self.test_point_batch_size],
            )

        # Get queue and fill up ones
        for i in range(1, dimf + 1):
            for j in range(1, dimf + 1):
                for k in range(1, dimf + 1):
                    maxv = np.max(
                        frame_flag[i - 1 : i + 2, j - 1 : j + 2, k - 1 : k + 2]
                    )
                    minv = np.min(
                        frame_flag[i - 1 : i + 2, j - 1 : j + 2, k - 1 : k + 2]
                    )
                    if maxv != minv:
                        queue.append((i, j, k))
                    elif maxv == 1:
                        x_coords = self.cell_x + (i - 1) * dimc
                        y_coords = self.cell_y + (j - 1) * dimc
                        z_coords = self.cell_z + (k - 1) * dimc
                        model_float[x_coords + 1, y_coords + 1, z_coords + 1] = 1.0

        cell_batch_size = dimc**3
        cell_batch_num = int(self.test_point_batch_size / cell_batch_size)
        assert cell_batch_num > 0

        # Run queue of non-empty cells
        while len(queue) > 0:
            batch_num = min(len(queue), cell_batch_num)
            point_list = []
            cell_coords = []
            for i in range(batch_num):
                point = queue.pop(0)
                point_list.append(point)
                cell_coords.append(
                    self.cell_coords[point[0] - 1, point[1] - 1, point[2] - 1]
                )
            cell_coords = np.concatenate(cell_coords, axis=0)
            cell_coords = np.expand_dims(cell_coords, axis=0)
            cell_coords = torch.from_numpy(cell_coords)
            cell_coords = cell_coords.to(self.device)
            _, model_out_batch_ = self.IMNet(None, z, cell_coords)
            model_out_batch = model_out_batch_.detach().cpu().numpy()[0]
            for i in range(batch_num):
                point = point_list[i]
                model_out = model_out_batch[
                    i * cell_batch_size : (i + 1) * cell_batch_size, 0
                ]
                x_coords = self.cell_x + (point[0] - 1) * dimc
                y_coords = self.cell_y + (point[1] - 1) * dimc
                z_coords = self.cell_z + (point[2] - 1) * dimc
                model_float[x_coords + 1, y_coords + 1, z_coords + 1] = model_out

                if np.max(model_out) > self.sampling_threshold:
                    for i in range(-1, 2):
                        pi = point[0] + i
                        if pi <= 0 or pi > dimf:
                            continue
                        for j in range(-1, 2):
                            pj = point[1] + j
                            if pj <= 0 or pj > dimf:
                                continue
                            for k in range(-1, 2):
                                pk = point[2] + k
                                if pk <= 0 or pk > dimf:
                                    continue
                                if frame_flag[pi, pj, pk] == 0:
                                    frame_flag[pi, pj, pk] = 1
                                    queue.append((pi, pj, pk))
        return model_float

    def optimize_mesh(self, vertices, z, iteration=3):
        """
        Optimize mesh vertices.
        May introduce foldovers.
        """
        new_vertices = np.copy(vertices)

        new_vertices_ = np.expand_dims(new_vertices, axis=0)
        new_vertices_ = torch.from_numpy(new_vertices_)
        new_vertices_ = new_vertices_.to(self.device)
        _, new_v_out_ = self.IMNet(None, z, new_vertices_)
        new_v_out = new_v_out_.detach().cpu().numpy()[0]

        for iter in range(iteration):
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    for k in [-1, 0, 1]:
                        if i == 0 and j == 0 and k == 0:
                            continue
                        offset = np.array([[i, j, k]], np.float32) / (
                            self.real_size * 6 * 2**iter
                        )
                        current_vertices = vertices + offset
                        current_vertices_ = np.expand_dims(current_vertices, axis=0)
                        current_vertices_ = torch.from_numpy(current_vertices_)
                        current_vertices_ = current_vertices_.to(self.device)
                        _, current_v_out_ = self.IMNet(None, z, current_vertices_)
                        current_v_out = current_v_out_.detach().cpu().numpy()[0]
                        keep_flag = abs(current_v_out - self.sampling_threshold) < abs(
                            new_v_out - self.sampling_threshold
                        )
                        keep_flag = keep_flag.astype(np.float32)
                        new_vertices = current_vertices * keep_flag + new_vertices * (
                            1 - keep_flag
                        )
                        new_v_out = current_v_out * keep_flag + new_v_out * (
                            1 - keep_flag
                        )
            vertices = new_vertices

        return vertices

    def decoder(self, z_batch, num_points=4096, sampling_factor=16):
        all_samples = np.empty([z_batch.shape[0], num_points, 3], dtype=np.float32)
        for i, z in enumerate(z_batch):
            model_float = self.z2voxel(z)
            vertices, triangles = mcubes.marching_cubes(
                model_float, self.sampling_threshold / sampling_factor
            )
            vertices = (vertices.astype(np.float32) - 0.5) / self.real_size - 0.5
            triangles = triangles.astype(np.int32)

            # Sample pointcloud
            pc_sample = sample_pc_random(
                vertices, triangles, num_points, check_normalize=False
            )
            all_samples[i] = pc_sample
        return torch.tensor(all_samples)


def load_imnet(ae_path):
    """
    Load IMNet Autoencoder.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    imnet = IMNetAE(sample_vox_size=64, device=device)

    # Load checkpoint
    imnet.IMNet.load_state_dict(torch.load(ae_path))
    imnet.IMNet = imnet.IMNet.eval()

    return imnet
