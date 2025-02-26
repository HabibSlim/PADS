"""
Training and evaluation functions for the PartQueries-conditioned diffusion model.
"""

import math
import sys

import torch
import torch.distributed as dist
import util.misc as misc
import util.s2vs as s2vs
import util.lr_sched as lr_sched
import wandb

PRINT_FREQ = 50


def compute_metrics(outputs, labels, threshold):
    """
    Compute intersection over union and accuracy metrics.
    """
    pred = torch.zeros_like(outputs)
    pred[outputs >= threshold] = 1

    accuracy = (pred == labels).float().sum(dim=1) / labels.shape[1]
    accuracy = accuracy.mean()

    intersection = (pred * labels).sum(dim=1)
    union = (pred + labels).gt(0).sum(dim=1) + 1e-5

    iou = intersection * 1.0 / union
    iou = iou.mean()

    # Print raw values
    # print("Outputs shape:", outputs.shape)
    # print("Labels shape:", labels.shape)
    # print("Sample outputs:", outputs[0, :10])
    # print("Sample labels:", labels[0, :10])

    # Check thresholding
    # print("Predictions before threshold:", outputs[0, :10])
    # print("Unique outputs values:", torch.unique(outputs))
    # print("Predictions after threshold:", pred[0, :10])
    # print("Unique prediction values:", torch.unique(pred))
    # print("Unique label values:", torch.unique(labels))

    # Debug IOU components
    # print("Intersection values:", intersection)
    # print("Union values:", union)
    # print("Raw IOU before mean:", iou)

    # Check for imbalance
    # print("Positive predictions:", (pred == 1).float().mean())
    # print("Positive labels:", (labels == 1).float().mean())

    return accuracy, iou


def forward_pass(args, model, data_tuple, criterion):
    """
    Compute a single forward pass of the model.
    """
    device = model.device

    # Unpack data dict
    part_points, part_bbs, occ_points, occ_labels, _ = data_tuple.values()

    # Move data to device
    part_points = part_points.to(device, non_blocking=True)
    part_bbs = part_bbs.to(device, non_blocking=True)

    occ_points = occ_points.to(device, non_blocking=True)
    occ_labels = occ_labels.to(device, non_blocking=True)

    # Forward pass
    outputs = model(part_points, part_bbs, occ_points)

    # Compute the KL loss
    if "kl" in outputs:
        loss_kl = outputs["kl"]
        loss_kl = torch.sum(loss_kl) / loss_kl.shape[0]
        loss_kl = args.kl_weight * loss_kl
    else:
        loss_kl = torch.tensor(0.0).to(device)

    # Chunk the outputs and labels
    outputs = outputs["logits"].squeeze()
    outputs = torch.where(torch.isnan(outputs), torch.zeros_like(outputs), outputs)

    occ_labels = occ_labels.float()
    vol_outs, near_outs = outputs.chunk(2, dim=1)
    vol_labels, near_labels = occ_labels.chunk(2, dim=1)

    # Show 100 first occupancy points
    # if args.global_rank == 0:
    #     vol_points, near_points = occ_points.chunk(2, dim=1)
    # print("Vol points shape:", occ_points.shape)
    # print("First 100 occupancy points:", occ_points[0, :100])
    # print("First 100 occupancy labels:", occ_labels[0, :100])
    # print("First 100 volume points:", vol_points[0, :100])
    # print("First 100 volume labels:", vol_labels[0, :100])
    # print("First 100 near points:", near_points[0, :100])
    # print("First 100 near labels:", near_labels[0, :100])

    # Compute the near-surface loss
    loss_near = torch.tensor(0.0).to(device)
    if args.near_weights is not None:
        """
        In this case occupancies are split like follows
        With N the total number of query points
        these take weights from near_weights array.
        """
        # Compute the near-surface loss
        for (k, near_weight), occ_chunk, occ_labels in zip(
            enumerate(args.near_weights),
            near_outs.chunk(2, dim=1),
            near_labels.chunk(2, dim=1),
        ):
            loss_near += near_weight * criterion(occ_chunk, occ_labels)
    else:
        loss_near = args.default_near_weights * criterion(near_outs, near_labels)

    # Add these checks before the loss calculation
    # if args.rank == 0:
    #     print("Vol outs shape:", vol_outs.shape)
    #     print(
    #         "Vol outs min/max:", torch.min(vol_outs).item(), torch.max(vol_outs).item()
    #     )
    #     print("Contains NaN:", torch.isnan(vol_outs).any().item())
    #     print("Contains inf:", torch.isinf(vol_outs).any().item())
    #     # Print a small sample of values
    #     print("Sample values:", vol_outs[:5].detach().cpu().numpy())

    # Compute the volume loss
    loss_vol = args.vol_weight * criterion(vol_outs, vol_labels)

    # Compute metrics for volume samples only
    if args.global_rank == 0:
        acc, iou = compute_metrics(vol_outs, vol_labels, threshold=0.0)
    else:
        acc, iou = torch.tensor(0.0).to(device), torch.tensor(0.0).to(device)

    dist.barrier()
    print("Losses:", loss_near, loss_vol, loss_kl)

    return {
        "loss_near": loss_near,
        "loss_vol": loss_vol,
        "loss_kl": loss_kl,
        "acc": acc,
        "iou": iou,
    }


def train_one_epoch(
    args,
    model,
    data_loader,
    optimizer,
    epoch,
    loss_scaler,
    max_norm=5.0,
):
    """
    Train the model for one epoch.
    """
    model.train(True)
    torch.autograd.set_detect_anomaly(True)

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Epoch: [{}]".format(epoch)

    accum_iter = args.accum_iter
    optimizer.zero_grad()

    criterion = torch.nn.BCEWithLogitsLoss()

    for data_step, data_tuple in enumerate(
        metric_logger.log_every(data_loader, PRINT_FREQ, header)
    ):
        # Per-iteration (instead of per epoch) lr scheduler
        if data_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_step / len(data_loader) + epoch, args
            )

        try:
            # Computing loss
            output = forward_pass(
                args=args, model=model, data_tuple=data_tuple, criterion=criterion
            )
            total_loss = output["loss_vol"] + output["loss_near"] + output["loss_kl"]
            loss_value = total_loss.item()

            # Backward pass
            total_loss /= accum_iter
            loss_scaler(
                total_loss,
                optimizer,
                clip_grad=max_norm,
                parameters=model.parameters(),
                create_graph=False,
                update_grad=(data_step + 1) % accum_iter == 0,
            )
        except RuntimeError as e:
            print(e)
            # Print the sampler properties
            print("Last seen group = ", data_loader.batch_sampler.last_batch_group)
            print()
            raise

        if (data_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        torch.cuda.synchronize()

        # Log the losses and metrics
        loss_update = {
            "loss_vol": float(output["loss_vol"]),
            "loss_near": float(output["loss_near"]),
            "loss_kl": float(output["loss_kl"]),
            "acc": float(output["acc"]),
            "iou": float(output["iou"]),
        }
        metric_logger.update(**loss_update)

        # Log the losses to wandb
        if args.global_rank == 0:
            wandb.log(
                {
                    "data_step": epoch * len(data_loader) + data_step,
                    "loss_vol_batch": float(output["loss_vol"]),
                    "loss_near_batch": float(output["loss_near"]),
                    "loss_kl_batch": float(output["loss_kl"]),
                    "acc_batch": float(output["acc"]),
                    "iou_batch": float(output["iou"]),
                }
            )

        dist.barrier()
        if args.global_rank == 0 and args.debug_run:
            print(
                "Rank [",
                args.global_rank,
                "] Step [",
                data_step,
                "]",
                "Number of Parts [",
                data_tuple["part_points"].shape[1],
                "]",
            )

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print()

    # Print metric logger keys
    print("Averaged stats:", metric_logger)
    if args.global_rank == 0:
        wandb.log(
            {
                "epoch": epoch,
                "loss_vol": float(metric_logger.loss_vol.global_avg),
                "loss_near": float(metric_logger.loss_near.global_avg),
                "loss_kl": float(metric_logger.loss_kl.global_avg),
                "acc": float(metric_logger.acc.global_avg),
                "iou": float(metric_logger.iou.global_avg),
            }
        )
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.inference_mode()
def extract_mesh(model, data_tuple, grid_density):
    """
    Extract mesh from occupancies predictions using volume samples only.
    Args:
        model: The model
        data_tuple: Dictionary containing:
            - part_points: (B=1, N, 3) tensor of point clouds
            - part_bbs: (B=1, P, 6) tensor of bounding boxes
            - model_ids: List containing a single model ID
        grid_density: Resolution of the grid
    Returns:
        str: Path to saved mesh file
    """
    device = model.device

    # Unpack and move data to device - note we expect B=1
    part_points, part_bbs = data_tuple["part_points"], data_tuple["part_bbs"]
    assert (
        part_points.shape[0] == 1
    ), f"Expected batch size 1, got {part_points.shape[0]}"

    if not part_points.device == device:
        part_points = part_points.to(device, non_blocking=True)
        part_bbs = part_bbs.to(device, non_blocking=True)

    # Generate a 3D grid of points
    grid_queries = s2vs.get_grid(grid_density=grid_density, grid_range=0.8).to(
        device
    )  # (G^3, 3)

    # Forward pass
    outputs = model(part_points, part_bbs, grid_queries)  # Add batch dim to grid
    outputs = outputs["logits"].squeeze(0)  # Remove batch dim from output

    # Extract mesh
    mesh = s2vs.reconstruct_mesh(outputs, grid_density=grid_density)

    # Save mesh to /tmp/
    mesh_file = f"/tmp/mesh_{data_tuple['model_ids'][0]}.obj"
    mesh.export(mesh_file)

    return mesh_file


@torch.inference_mode()
def evaluate(
    args,
    model,
    data_loader,
    epoch,
):
    """
    Evaluate the model on the validation set.
    """
    model.eval()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Validation:"

    criterion = torch.nn.BCEWithLogitsLoss()

    data_seen = False
    for data_step, data_tuple in enumerate(
        metric_logger.log_every(data_loader, PRINT_FREQ, header)
    ):
        # Computing forward pass and metrics
        output = forward_pass(
            args=args, model=model, data_tuple=data_tuple, criterion=criterion
        )

        # Update running metrics
        loss_update = {
            "loss_vol": float(output["loss_vol"]),
            "loss_near": float(output["loss_near"]),
            "loss_kl": float(output["loss_kl"]),
            "acc": float(output["acc"]),
            "iou": float(output["iou"]),
        }
        metric_logger.update(**loss_update)

        data_seen = True

    assert data_seen, "No data seen in evaluation loop"

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()

    # Print metric logger keys
    print("Validation stats:", metric_logger)
    torch.cuda.empty_cache()

    # Also extract occupancy predictions for visualization
    if args.global_rank == 0 and epoch % 10 == 0:
        with torch.no_grad():
            # Calculate number of samples to extract (25% of batch)
            batch_size = len(data_tuple["model_ids"])
            num_samples = max(1, int(0.25 * batch_size))

            # Extract meshes for multiple samples
            for idx in range(num_samples):
                # Create single-sample data dictionary, preserving batch dimension
                sample_data = {
                    "part_points": data_tuple["part_points"][
                        idx : idx + 1
                    ],  # Keep batch dim
                    "part_bbs": data_tuple["part_bbs"][idx : idx + 1],  # Keep batch dim
                    "model_ids": [data_tuple["model_ids"][idx]],
                }
                mesh_file = extract_mesh(
                    model, sample_data, grid_density=args.grid_density
                )

                # Log the mesh file to wandb
                if args.debug_run:
                    mesh_file = misc.gen_dummy_mesh(mesh_file)
                wandb.log(
                    {
                        f"mesh_gen_{idx}_{epoch}": wandb.Object3D.from_file(
                            open(mesh_file)
                        ),
                    }
                )

    # Log validation metrics to wandb
    if args.global_rank == 0:
        wandb.log(
            {
                "epoch": epoch,
                "val_loss_vol": float(metric_logger.loss_vol.global_avg),
                "val_loss_near": float(metric_logger.loss_near.global_avg),
                "val_loss_kl": float(metric_logger.loss_kl.global_avg),
                "val_acc": float(metric_logger.acc.global_avg),
                "val_iou": float(metric_logger.iou.global_avg),
            }
        )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
