"""
Evaluation utility functions.
"""
import argparse
import warnings
import torch

import subprocess
import models.mlp_mapper as mlp_mapper

from PIL import Image
from pdflatex import PDFLaTeX
from IPython.display import Image as ImageDisplay
from util.datasets import build_shape_surface_occupancy_dataset
from eval.chain_sampler import ChainSampler

warnings.filterwarnings("ignore")


def init_exps(model_name, model_path, ae_model):
    """
    Initialize the latent space mapper and args.
    """
    # Set dummy arg string to debug the parser
    call_string = """--ae-latent-dim 256 \
        --text_model_name bert-base-uncased \
        --dataset graphedits_chained \
        --data_path /ibex/user/slimhy/ShapeWalk/ \
        --data_type release_chained \
        --num_workers 8 \
        --model %s \
        --resume %s \
        --resume_full_weights \
        --device cuda \
        --fetch_keys \
        --use_embeds \
        --alt_ae_embeds %s \
        --seed 0""" % (model_name, model_path, ae_model)

    # Parse the arguments
    args = get_args_parser()
    args = args.parse_args(call_string.split())
    args.use_clip = "clip" in args.text_model_name
    device = torch.device(args.device)

    model = mlp_mapper.__dict__[args.model](use_linear_proj=not args.use_clip)
    model.to(device)

    # Load the checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])

    return args, model, device


def get_loader(args, batch_size, chain_length):
    """
    Get the data loader for chained evaluation.
    """
    args.batch_size = batch_size
    args.chain_length = chain_length

    dataset_val = build_shape_surface_occupancy_dataset("val", args=args)
    chain_sampler = ChainSampler(
        dataset_val, batch_size=args.batch_size, chain_length=args.chain_length
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=chain_sampler,
    )

    return data_loader_val


def get_args_parser():
    parser = argparse.ArgumentParser("Performing Chained Eval", add_help=False)

    # Model parameters
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size per GPU"
        "(effective batch size is batch_size * accum_iter * # gpus",
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
    parser.add_argument("--ae_pth", help="Autoencoder checkpoint")
    parser.add_argument("--point_cloud_size", default=2048, type=int, help="input size")
    parser.add_argument(
        "--fetch_keys",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--use_clip",
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
        choices=["graphedits", "graphedits_chained"],
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
        "--chain_length",
        default=None,
        type=int,
        help="length of chains to load",
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
    parser.add_argument(
        "--alt_ae_embeds",
        type=str,
        default=None,
        help="Alternative autoencoder embeddings to use",
    )
    parser.add_argument(
        "--ft_bert",
        action="store_true",
        default=False,
        help="Also fine-tune the BERT model",
    )
    parser.add_argument(
        "--model",
        type=str,
        metavar="MODEL",
    )
    parser.add_argument("--resume", default="", help="Resume from checkpoint")
    parser.add_argument(
        "--resume_full_weights",
        action="store_true",
        default=False,
        help="Resume the full model weights with the EDM wrapper",
    )

    return parser


def display_latex(content, crop_zone=(0, 0, 1000, 1000), hspace=29):
    """
    Render LaTeX content in the Jupyter Notebook.
    """
    page_header = r"""
    \documentclass{article}
    \usepackage[paper=portrait,pagesize]{typearea}
    \pagenumbering{gobble}
    % CUSTOM PACKAGES
    % ------------------------------------------
    \usepackage[dvipsnames]{xcolor}
    \usepackage{amsmath, amsfonts}
    \usepackage{fontawesome5}
    \usepackage{booktabs}
    \usepackage{multirow}
    \usepackage{colortbl}
    \usepackage{xcolor}
    \usepackage{lscape}

    % "Yes" icon
    \newcommand{\icoyes}{\textcolor{ForestGreen}{\faIcon{check-circle}}}
    % "No" icon
    \newcommand{\icono}{\textcolor{Red}{\faIcon{times-circle}}}

    \begin{document}
        \KOMAoptions{paper=landscape,pagesize}
        \recalctypearea
        \begin{table}[h!]
            \vspace{-1em}
            \centering
            \renewcommand{\arraystretch}{1.5}
            \resizebox{1.7\textwidth}{!}{%
    """
    page_hspace = " \hspace{-%dem} " % hspace
    page_footer = r"""
            }
            \vspace{1em}
        \end{table}
    \end{document}
    """
    content = page_header + page_hspace + content + page_footer
    content_binary = content.encode("utf-8")
    pdfl = PDFLaTeX.from_binarystring(content_binary, jobname="/tmp/__temp__")
    pdf, log, completed_process = pdfl.create_pdf(
        keep_pdf_file=True, keep_log_file=True
    )

    # Convert pdf to png
    subprocess.call(["pdftoppm", "-png", "/tmp/__temp__.pdf", "/tmp/__temp__"])
    # Crop image
    img = Image.open("/tmp/__temp__-1.png")
    img = img.crop(crop_zone)
    img.save("/tmp/__temp__.png")

    # Display image
    return ImageDisplay(filename="/tmp/__temp__.png")
