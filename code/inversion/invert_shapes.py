"""
Invert a set of input shapes.
"""

import argparse
import torch

import util.misc as misc
import util.s2vs as s2vs

from schedulefree import AdamWScheduleFree
from datasets.shapeloaders import SingleManifoldDataset
from inversion.optimize import optimize_latents, refine_latents
from inversion.evaluate import evaluate_reconstruction


ACTIVE_CLASS = "chair"
OBJ_DIR = "/ibex/user/slimhy/PADS/data/obj_manifold/"
N_POINTS = 2**21
MAX_POINTS = 2**21  # Maximum number of points for a single batch on a A100 GPU


"""
Main routine functions. 
"""


def get_args_parser():
    """
    Parsing input arguments.
    """
    parser = argparse.ArgumentParser("Extracting Features", add_help=False)

    # Model parameters
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument(
        "--text_model_name",
        type=str,
        help="Text model name to use",
    )
    parser.add_argument(
        "--ae",
        type=str,
        metavar="MODEL",
        help="Name of autoencoder",
    )
    parser.add_argument(
        "--ae-latent-dim",
        type=int,
        default=512 * 8,
        help="AE latent dimension",
    )
    parser.add_argument("--ae_pth", required=True, help="Autoencoder checkpoint")
    parser.add_argument("--point_cloud_size", default=2048, type=int, help="input size")
    parser.add_argument(
        "--fetch_keys",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--use_embeds",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--intensity_loss",
        action="store_true",
        default=False,
        help="Contrastive edit intensity loss using ground-truth labels.",
    )

    # Dataset parameters
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["graphedits"],
        help="dataset name",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="dataset path",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        help="dataset type",
    )
    parser.add_argument(
        "--max_edge_level",
        default=None,
        type=int,
        help="maximum edge level to use",
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_workers", default=60, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )

    # Data split parameters
    parser.add_argument("--obj-id-list", type=str, help="List of object IDs to process")

    # Hyperparameters

    return parser


def initialize_latents(ae, obj_id):
    """
    Get the initial latents for the optimization.
    """
    orig_dataset = SingleManifoldDataset(
        OBJ_DIR, ACTIVE_CLASS, ae.num_inputs, normalize=False, sampling_method="surface"
    )
    surface_points, _ = next(orig_dataset[obj_id])
    init_latents = s2vs.encode_pc(ae, surface_points).detach()

    return init_latents


def main(args):
    """
    Main function.
    """
    # Set device and seed
    device = torch.device(args.device)
    misc.set_all_seeds(args.seed)
    torch.backends.cudnn.benchmark = True

    # Instantiate autoencoder
    ae = s2vs.load_model(args.ae, args.ae_pth, device, torch_compile=True)
    ae = ae.eval()

    for obj_id in args.obj_id_list:
        # Initialize the latents
        init_latents = initialize_latents(ae, obj_id)

        # Instantiate the shape dataset
        shape_dataset = SingleManifoldDataset(
            OBJ_DIR,
            ACTIVE_CLASS,
            N_POINTS,
            normalize=False,
            sampling_method="volume+near_surface",
            contain_method="occnets",
            decimate=True,
            sample_first=False,
        )

        # Optimize the latents
        optimized_latents = optimize_latents(
            ae,
            shape_dataset,
            init_latents,
            acc_steps=N_POINTS // MAX_POINTS if N_POINTS > MAX_POINTS else 1,
            max_iter=50,
            optimizer=AdamWScheduleFree,
        )

        # Refine the latents in a second stage
        refined_latents = refine_latents(
            ae,
            shape_dataset.mesh,
            optimized_latents,
            acc_steps=N_POINTS // MAX_POINTS if N_POINTS > MAX_POINTS else 1,
            max_iter=50,
            optimizer=AdamWScheduleFree,
        )

        # Decode the optimized latents
        rec_mesh = s2vs.decode_latents(
            ae, misc.d_GPU(refined_latents), grid_density=256, batch_size=128**3
        )

        # Evaluate the final reconstruction
        eval_metrics = evaluate_reconstruction(
            shape_dataset.mesh, rec_mesh, optimized_latents, query_method="occnets"
        )
        eval_metrics["obj_id"] = obj_id


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    args.obj_id_list = args.obj_id_list.split(",")
    main(args)
