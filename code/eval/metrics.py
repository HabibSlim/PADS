"""
Compute evaluation metrics.
"""

import point_cloud_utils as pcu
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from losses.chamfer import chamfer_loss
from eval.apply_edits import apply_iterated_edits
from eval.sampling_util import sample_pc_random
from eval.utils import get_loader
from models.dgcnn import DGCNN
import util.misc as misc


def load_node_mesh(data_path, node_id):
    """
    Load a mesh corresponding to a specific node.
    """
    data_path = data_path + "obj/"
    mesh_path = data_path + node_id + ".obj"
    v, f = pcu.load_mesh_vf(mesh_path)
    return v, f


def sample_mesh(data_path, node_id, n_points):
    """
    Sample pointcloud from a mesh surface.
    """
    v, f = load_node_mesh(data_path, node_id)
    return sample_pc_random(v, f, n_points)


def load_pointcloud(data_path, node_id):
    """
    Load a pointcloud corresponding to a specific node.
    """
    data_path = data_path + "pc/"
    pc_path = data_path + node_id + ".npy"
    return np.load(pc_path)


"""
Defining all metrics.
"""


def l2_dist(x_gt, x_edited):
    """
    Compute the L2 distance between two latents.
    """
    return torch.norm(x_edited - x_gt, p=2)


def chamfer_reconstructed(pc_gt, pc_pred):
    """
    Compute the chamfer distance between the reconstructed edited and gt pointclouds.
    """
    return chamfer_loss(pc_gt, pc_pred, reduction="mean").mean()


def iou_occupancies(pred_occ, gt_occ):
    """
    Compute the IoU between predicted and ground-truth occupancy vectors.
    """
    iou = (np.logical_and(pred_occ, gt_occ).sum() + 1e-5) / (
        np.logical_or(pred_occ, gt_occ).sum() + 1e-5
    )
    return iou


def chamfer_real__single(
    p_edited, node_gt, transform, data_path, n_samples=8, resample=False
):
    """
    Worker function for chamfer_real.
    """
    if type(transform) == tuple:
        t_rec, t_mesh = transform
    else:
        t_rec = lambda x: x
        t_mesh = transform
    n_input = p_edited.shape[0]

    p_edited = p_edited.unsqueeze(0).float().cuda()
    p_edited = t_rec(p_edited)

    # Sample n_samples x n_input points
    if resample:
        p = sample_mesh(data_path, node_gt, n_samples * n_input)
    else:
        p = load_pointcloud(data_path, node_gt)
        # Downsample if necessary
        if p.shape[0] > n_samples * n_input:
            np.random.shuffle(p)
            p = p[: n_samples * n_input, :]

    # Randomly split points into batches of size n_input
    np.random.shuffle(p)
    n_batches = p.shape[0] // n_input
    p = p.reshape(n_batches, n_input, 3)

    # For each batch, compute average distance to the input pointcloud
    avg_dist = np.zeros(n_batches)
    with torch.no_grad():
        for i in range(n_batches):
            # Transform the batch
            p[i] = t_mesh(p[i])
            p_c = torch.from_numpy(p[i]).float().cuda().unsqueeze(0)

            # Compute chamfer distance to input pointcloud
            cd_dist = chamfer_loss(p_c, p_edited, reduction="mean").mean()
            avg_dist[i] = cd_dist.item()

    # Compute the average distance and standard deviation
    avg_dist = np.min(avg_dist)

    return avg_dist


def chamfer_real(p_edited, node_gt, transform, data_path, n_samples=8):
    """
    Compute the chamfer distance between the edited pointcloud and a sampled pointcloud from the gt mesh.
    """
    avg_dist = 0.0
    for i in range(p_edited.shape[0]):
        edited_node_id = node_gt[i].split("_")[1]
        avg_dist += chamfer_real__single(
            p_edited[i], edited_node_id, transform, data_path, n_samples
        )
    avg_dist /= p_edited.shape[0]
    return avg_dist


def dgcnn_ft_dist__single(
    encoder, z_edited, node_gt, transform, data_path, n_samples=8
):
    """
    Compute the feature distance between the edited pointcloud and a sampled pointcloud from the gt mesh.
    """
    n_input = encoder.config.nrof_points

    # Sample 2^n points
    p = sample_mesh(data_path, node_gt, n_samples * n_input)

    # Randomly split points into batches of size n_input
    np.random.shuffle(p)
    n_batches = p.shape[0] // n_input
    p = p.reshape(n_batches, n_input, 3)

    # For each batch, apply the transform
    with torch.no_grad():
        for i in range(n_batches):
            p[i] = transform(p[i])

    # Compute feature vector for all batches
    z_c = encoder.forward_features(torch.from_numpy(p).float().cuda())

    # For each batch, compute average distance to the input pointcloud
    avg_dist = np.zeros(n_batches)
    with torch.no_grad():
        for i in range(n_batches):
            z_c_i = z_c[i].unsqueeze(0)
            avg_dist[i] = torch.norm(z_edited - z_c_i, p=2).item()

    # Compute the average distance and standard deviation
    avg_dist = np.mean(avg_dist)

    return avg_dist


def dgcnn_ft_dist(encoder, p_edited, node_gt, transform, data_path, n_samples=8):
    """
    Compute the feature distance between the edited pointcloud and a sampled pointcloud from the gt mesh.
    """
    avg_dist = 0.0
    for i in range(p_edited.shape[0]):
        edited_node_id = node_gt[i].split("_")[1]
        z_edited = encoder.forward_features(p_edited[i].unsqueeze(0))
        avg_dist += dgcnn_ft_dist__single(
            encoder, z_edited, edited_node_id, transform, data_path, n_samples
        )
    avg_dist /= p_edited.shape[0]
    return avg_dist


"""
Running the experiments and evaluating.
"""


def get_metrics(
    args, model, ae_model, encoder, data_loader, drop_n, pc_t, oracle_type="none"
):
    """
    Get the metrics for chained evaluation.
    """
    metric_meter = misc.MetricLogger()

    with torch.no_grad():
        chain_count = 0
        for batch_k, (chain_ids, edit_keys, node_a, node_b, text_embeds) in enumerate(
            data_loader
        ):
            if batch_k == len(data_loader) - 1 - drop_n:
                break
            if chain_count == 0:
                prev_node = node_a

            # Apply the edits
            (p_b_pred, p_b), (x_b_pred, x_b) = apply_iterated_edits(
                model,
                ae_model,
                embeds_a=prev_node,
                embeds_b=node_b,
                embeds_text=text_embeds,
                oracle_type=oracle_type,
            )

            # Compute average pairwise L2 distance in feature space
            l2_distance = l2_dist(x_b, x_b_pred)

            # Compute average pairwise reconstructed CD
            cd_dist_reco = chamfer_reconstructed(p_b, p_b_pred)

            # Compute average pairwise real CD
            cd_dist_real = chamfer_real(
                p_edited=p_b_pred,
                node_gt=edit_keys,
                transform=pc_t,
                data_path=args.data_path,
                n_samples=8,
            )

            # Compute average pairwise L2 distance in DGCNN feature space
            # encoder, p_edited, node_gt, transform, data_path, n_samples=8
            # dgcnn_distance = dgcnn_ft_dist(
            #     encoder,
            #     p_edited=p_b_pred,
            #     node_gt=edit_keys,
            #     transform=pc_t,
            #     data_path=args.data_path,
            #     n_samples=4,
            # )

            # Log all metrics
            metric_meter.update(
                avg_l2_dist=l2_distance.item(),
                avg_cd_dist_reco=cd_dist_reco.item(),
                avg_cd_dist_real=cd_dist_real.item(),
                # avg_dgcnn_ft_dist=dgcnn_distance.item(),
            )

            prev_node = x_b_pred

            # Log final chain metrics
            chain_count += 1
            if chain_count == args.chain_length or True:
                metric_meter.update(
                    final_l2_dist=l2_distance.item(),
                    final_cd_dist_reco=cd_dist_reco.item(),
                    final_cd_dist_real=cd_dist_real.item(),
                    # final_dgcnn_ft_dist=dgcnn_distance.item(),
                )
                chain_count = 0

    return metric_meter


def run_exps(args, model, ae_model, device, pc_transform, oracle_type="none"):
    """
    Run the chained evaluations for different chain lengths.
    """
    # Fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    args.fetch_keys = True

    # Instantiate the model
    model = model.eval().cuda()

    # Instantiate the DGCNN model
    encoder = DGCNN()
    encoder = encoder.eval().cuda()

    all_metrics = {
        "final_cd_dist_reco": [],
        "avg_cd_dist_reco": [],
        "final_cd_dist_real": [],
        "avg_cd_dist_real": [],
        "final_l2_dist": [],
        "avg_l2_dist": [],
        # "final_dgcnn_ft_dist": [],
        # "avg_dgcnn_ft_dist": [],
    }
    all_tuples = []
    for chain_length, batch_size, drop_n in [[10, 8, 0], [15, 4, 3], [20, 3, 0]]:
        data_loader_val = get_loader(
            args, batch_size=batch_size, chain_length=chain_length
        )
        metric_meter = get_metrics(
            args,
            model,
            ae_model,
            encoder,
            data_loader_val,
            drop_n,
            pc_transform,
            oracle_type,
        )

        # Print the results
        for k, v in metric_meter.meters.items():
            if "cd" in k:
                all_metrics[k].append(v.global_avg * 10**4)
            elif "dgcnn" in k:
                all_metrics[k].append(v.global_avg * 10**3)
            else:
                all_metrics[k].append(v.global_avg)

        row_tuple = tuple(all_metrics[k][-1] for k in all_metrics.keys())
        all_tuples.append(row_tuple)

    full_results = {
        "mean": {k: np.mean(v) for k, v in all_metrics.items()},
        "tuples": all_tuples,
    }
    return full_results
