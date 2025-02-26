"""
Visualization utilities for occupancy queries.
"""

import plotly.graph_objects as go
import numpy as np
import torch


def unpack_bb_data(bb_data):
    """
    Load part bounding boxes for a given model ID.
    /!\ Loses information about which part is which.
    """
    bb_data = {k: v for k, v in bb_data}
    bb_data = {k: np.array(v.vertices) for k, v in bb_data.items()}
    return np.concatenate([v for v in bb_data.values()], axis=0)


def viz_part_pointcloud(part_points):
    """
    Visualize a part point cloud.

    Args:
        part_points: Part point cloud of shape [N_points, 3]

    Returns:
        Plotly figure with part points
    """
    # Convert to numpy if it's a tensor
    if isinstance(part_points, torch.Tensor):
        part_points = part_points.numpy()

    # Create figure
    fig = go.Figure()

    # Add part points
    fig.add_trace(
        go.Scatter3d(
            x=part_points[:, 0],
            y=part_points[:, 1],
            z=part_points[:, 2],
            mode="markers",
            marker=dict(size=2, color="blue", opacity=0.7),
            name="Part Points",
        )
    )

    # Set layout
    fig.update_layout(
        title="Part Visualization",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            camera=dict(
                up=dict(x=0, y=1, z=0),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5),
            ),
        ),
    )

    return fig


def viz_queries(data_tuple, sample_idx=0, unpack_bb=True):
    """
    Visualize query points colored by occupancy and bounding box corners.

    Args:
        data_tuple: Dictionary containing query points, labels and bounding boxes
        sample_idx: Index of the sample to visualize
    """
    # Get query points and labels
    query_points = data_tuple["query_points"][sample_idx].cpu().numpy()  # [1, N, 3]
    query_labels = data_tuple["query_labels"][sample_idx].cpu().numpy()  # [N,]

    query_points = query_points.squeeze(0)  # [N, 3]

    print(query_points.shape, query_labels.shape)

    # Get bounding box corners
    bbs = data_tuple["part_bbs"][sample_idx]  # List
    if unpack_bb:
        bb_points = unpack_bb_data(bbs)  # [P, 8, 3]
    else:
        bb_points = bbs.reshape(-1, 3)
    # bb_points = bbs.reshape(-1, 3)  # [P*8, 3]

    # Split points into volume and near-surface if needed
    N = len(query_points)

    # Split query points by label
    inside_mask = query_labels == 1
    inside_points = query_points[inside_mask]
    outside_points = query_points[~inside_mask]

    fig = go.Figure(
        data=[
            # Inside query points (red)
            go.Scatter3d(
                x=inside_points[:, 0] if len(inside_points) > 0 else [],
                y=inside_points[:, 1] if len(inside_points) > 0 else [],
                z=inside_points[:, 2] if len(inside_points) > 0 else [],
                mode="markers",
                marker=dict(size=2, color="red", opacity=0.8),
                name="Inside Points",
            ),
            # Outside query points (green)
            go.Scatter3d(
                x=outside_points[:, 0] if len(outside_points) > 0 else [],
                y=outside_points[:, 1] if len(outside_points) > 0 else [],
                z=outside_points[:, 2] if len(outside_points) > 0 else [],
                mode="markers",
                marker=dict(size=2, color="green", opacity=0.8),
                name="Outside Points",
            ),
            # Bounding box corners (blue)
            go.Scatter3d(
                x=bb_points[:, 0],
                y=bb_points[:, 1],
                z=bb_points[:, 2],
                mode="markers",
                marker=dict(size=8, color="blue", symbol="square", opacity=1.0),
                name="Box Corners",
            ),
        ]
    )

    # Update title based on point type
    title = "Query Points and Box Corners"

    fig.update_layout(
        title=title,
        scene=dict(
            aspectmode="data", xaxis_title="X", yaxis_title="Y", zaxis_title="Z"
        ),
        showlegend=True,
        width=800,
        height=800,
    )

    return fig


def compute_occupancy_ratios(data_tuple, sample_idx=0):
    """
    Compute the ratio of 0s to 1s in occupancy labels.

    Args:
        data_tuple: Dictionary containing query labels
        sample_idx: Index of the sample to analyze

    Returns:
        dict: Dictionary containing counts and ratios
    """
    # Get query labels
    labels = data_tuple["query_labels"][sample_idx].cpu().numpy()  # [N]

    # Split points into volume and near-surface if needed
    N = len(labels)

    # Count 0s and 1s
    zeros = (labels == 0).sum()
    ones = (labels == 1).sum()
    total = len(labels)

    # Compute ratios
    ratio_zeros = zeros / total if total > 0 else 0
    ratio_ones = ones / total if total > 0 else 0
    zero_to_one_ratio = zeros / ones if ones > 0 else float("inf")

    return {
        "counts": {"zeros": int(zeros), "ones": int(ones), "total": int(total)},
        "ratios": {
            "zeros": float(ratio_zeros),
            "ones": float(ratio_ones),
            "zero_to_one": float(zero_to_one_ratio),
        },
    }


def print_occupancy_stats(data_tuple, sample_idx=0):
    """
    Print comprehensive occupancy statistics for all point types.

    Args:
        data_tuple: Dictionary containing query labels
        sample_idx: Index of the sample to analyze
    """
    # Compute stats for all point types
    stats = compute_occupancy_ratios(data_tuple, sample_idx)

    # Print results
    counts = stats["counts"]
    ratios = stats["ratios"]

    print(f"\nPoints Statistics:")
    print(f"Total points: {counts['total']}")
    print(f"Zeros: {counts['zeros']} ({ratios['zeros']:.2%})")
    print(f"Ones: {counts['ones']} ({ratios['ones']:.2%})")
    print(f"Ratio of zeros to ones: {ratios['zero_to_one']:.2f}")


# Modified visualization helper function to work with direct inputs
def convert_to_data_tuple(points, occs, bbs):
    """Convert raw data to the expected data_tuple format"""
    return {
        "query_points": [points],  # Wrap in list to match expected batch dimension
        "query_labels": [occs],
        "part_bbs": [bbs],
    }
