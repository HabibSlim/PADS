"""
Invert a set of input shapes.
"""

import argparse
import json
import torch
import os

import util.misc as misc
import util.s2vs as s2vs

from schedulefree import AdamWScheduleFree
from datasets.shapeloaders import SingleManifoldDataset
from inversion.optimize import optimize_latents, refine_latents
from inversion.evaluate import evaluate_reconstruction


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
        default=128,
        type=int,
        help="Batch size per GPU (this is the grid dimension, to be cubed)",
    )
    parser.add_argument(
        "--ae",
        type=str,
        metavar="MODEL",
        help="Name of autoencoder",
    )
    parser.add_argument(
        "--ae_latent_dim",
        type=int,
        default=512 * 8,
        help="AE latent dimension",
    )
    parser.add_argument("--ae_pth", required=True, help="Autoencoder checkpoint")
    parser.add_argument("--point_cloud_size", default=2048, type=int, help="input size")

    # CUDA parameters
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

    # Datasplit parameters
    parser.add_argument("--obj_id_list", type=str, help="List of object IDs to process")
    parser.add_argument("--max_id", type=int, help="Maximum object ID to process")
    parser.add_argument("--obj_dir", type=str, help="Path to object directory")
    parser.add_argument(
        "--active_class", type=str, help="Active object category to process"
    )
    parser.add_argument(
        "--n_points",
        type=int,
        help="Number of points to sample from the object surface (2^N)",
        default=21,
    )
    parser.add_argument(
        "--max_points",
        type=int,
        help="Maximum number of points to sample from the object surface (2^N)",
        default=21,
    )

    # Hyperparameters
    parser.add_argument("--sampling_method", type=str, help="Sampling method")
    parser.add_argument(
        "--near_surface_noise",
        type=float,
        help="Noise level for near-surface sampling",
    )
    parser.add_argument(
        "--squeeze_factor",
        type=float,
        help="Squeeze factor for face sampling",
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        help="Maximum number of iterations for each stage",
    )

    # Additional hyperparameters
    parser.add_argument(
        "--refresh_dist_every",
        type=int,
        help="Refresh the error distribution every N iterations",
    )

    # Logging
    parser.add_argument(
        "--log_dir",
        type=str,
        help="Logging directory",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        help="Name of the active configuration",
    )

    return parser


def initialize_latents(args, ae, obj_id):
    """
    Get the initial latents for the optimization.
    """
    orig_dataset = SingleManifoldDataset(
        args.obj_dir,
        args.active_class,
        ae.num_inputs,
        normalize=False,
        sampling_method="surface",
    )
    surface_points, _ = next(orig_dataset[obj_id])
    init_latents = s2vs.encode_pc(ae, surface_points).detach()
    gt_mesh = orig_dataset.get_mesh(obj_id)

    return init_latents, gt_mesh


def get_metrics(ae, gt_mesh, latents, batch_size):
    """
    Get the metrics for an input latent.
    """
    rec_mesh = s2vs.decode_latents(
        ae=ae,
        latent=misc.d_GPU(latents),
        grid_density=512,
        batch_size=batch_size,
    )
    metrics = evaluate_reconstruction(
        ae=ae,
        gt_mesh=gt_mesh,
        rec_mesh=rec_mesh,
        latents=latents,
        query_method="occnets",
    )

    return metrics


def main(args):
    """
    Main function.
    """
    # Parse input arguments
    if args.max_id is not None:
        args.obj_id_list = list(range(args.max_id))
    else:
        args.obj_id_list = [int(i) for i in args.obj_id_list.split(",")]
    args.n_points = 2**args.n_points
    args.max_points = 2**args.max_points
    args.refine_sampling_method = (
        "volume+near_surface_weighted"
        if "volume" in args.sampling_method
        else "near_surface_weighted"
    )
    args.batch_size = args.batch_size**3

    # Print input arguments
    print("Input args:\n", json.dumps(vars(args), indent=4, sort_keys=True))
    print("Job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    os.makedirs(args.log_dir, exist_ok=True)

    # Set device and seed
    device = torch.device(args.device)
    misc.set_all_seeds(args.seed)
    torch.backends.cudnn.benchmark = True

    # Instantiate autoencoder
    ae = s2vs.load_model(args.ae, args.ae_pth, device, torch_compile=True)
    ae = ae.eval()

    n_acc_steps = (
        args.n_points // args.max_points if args.n_points > args.max_points else 1
    )

    all_metrics = {}
    for obj_id in args.obj_id_list:
        print("Processing object: [%d]" % obj_id)
        # Initialize the latents, define the ground truth mesh
        init_latents, gt_mesh = initialize_latents(args, ae, obj_id)

        # Evaluate the initial reconstruction
        init_metrics = get_metrics(
            ae=ae,
            gt_mesh=gt_mesh,
            latents=init_latents,
            batch_size=args.batch_size,
        )

        # Instantiate the shape dataset
        shape_dataset = SingleManifoldDataset(
            args.obj_dir,
            args.active_class,
            args.n_points,
            near_surface_noise=args.near_surface_noise,
            normalize=False,
            sampling_method=args.sampling_method,
            contain_method="occnets",
            decimate=True,
            sample_first=False,
        )

        print("Starting first stage optimization for object: [%d]..." % obj_id)

        # Optimize the latents
        optimized_latents = optimize_latents(
            ae=ae,
            shape_dataset=shape_dataset,
            init_latents=init_latents,
            object_id=obj_id,
            accumulation_steps=n_acc_steps,
            max_iter=args.max_iter,
            optimizer=AdamWScheduleFree,
            lr=args.lr,
        )

        # Decode the optimized latents
        first_stage_metrics = get_metrics(
            ae=ae,
            gt_mesh=gt_mesh,
            latents=optimized_latents,
            batch_size=args.batch_size,
        )
        print("Done. Starting second stage optimization for object: [%d]..." % obj_id)

        # Refine the latents in a second stage
        refined_latents = refine_latents(
            ae=ae,
            gt_mesh=gt_mesh,
            init_latents=optimized_latents,
            num_points=args.n_points,
            best_loss=first_stage_metrics["depth_loss"],
            near_surface_noise=args.near_surface_noise,
            accumulation_steps=n_acc_steps,
            max_iter=args.max_iter,
            refresh_dist_every=args.refresh_dist_every,
            optimizer=AdamWScheduleFree,
            lr=args.lr,
            sampling_method=args.refine_sampling_method,
            squeeze_factor=args.squeeze_factor,
        )

        # Evaluate the final reconstruction
        final_metrics = get_metrics(
            ae=ae,
            gt_mesh=gt_mesh,
            latents=refined_latents,
            batch_size=args.batch_size,
        )

        # Log the results
        all_metrics[obj_id] = {
            "initial": init_metrics,
            "first_stage": first_stage_metrics,
            "final": final_metrics,
        }

        # Save the latents to disk
        # use torch.save
        torch.save(
            refined_latents,
            os.path.join(
                args.log_dir,
                "3DCoMPaT_latents",
                "latents_%s_%d.pt" % (args.active_class, obj_id),
            ),
        )

        print("Done. Object: [%d] processed." % obj_id)

    # Dump the results as a JSON
    misc.dump_json(
        all_metrics, args.log_dir, "inversion_results_%s.json" % args.config_name
    )

    print(
        "Inversion completed for class: [%s]" % args.active_class,
        " and objects: [%s]." % "|".join([str(i) for i in args.obj_id_list]),
    )


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
