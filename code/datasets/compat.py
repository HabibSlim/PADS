"""
Dataset iterators for 3DCoMPaT and ShapeNet.
"""

import json
import os
import numpy as np
import zipfile
from datasets.CoMPaT import compat3D
from datasets.metadata import (
    SHAPENET_CLASSES,
    COMPAT_CLASSES,
    COMPAT_TRANSFORMS,
    int_to_hex,
)

METADATA_DIR = "/ibex/user/slimhy/3DCoMPaT/3DCoMPaT-v2/metadata"
ZIP_SRC = "/ibex/user/slimhy/surfaces.zip"
ZIP_PATH = "/ibex/user/slimhy/3DCoMPaT/3DCoMPaT_ZIP.zip"


def shapenet_iterator(shape_cls):
    """
    ShapeNet iterator.
    """
    # List all files in the zip file
    with zipfile.ZipFile(ZIP_SRC, "r") as zip_ref:
        files = zip_ref.namelist()

        for file in files:
            if not file.startswith(SHAPENET_CLASSES[shape_cls]):
                continue
            if not file.endswith(".npz"):
                continue
            # Read a specific file from the zip file
            with zip_ref.open(file) as file:
                data = np.load(file)
                yield data["points"].astype(np.float32)


def compat_iterator(shape_cls, num_points):
    """
    3DCoMPaT iterator.
    """
    train_dataset = compat3D.ShapeLoader(
        zip_path=ZIP_PATH,
        meta_dir=METADATA_DIR,
        split="train",
        n_points=num_points,
        shuffle=True,
        seed=0,
        filter_class=COMPAT_CLASSES[shape_cls],
    )

    for shape_id, shape_label, pointcloud, point_part_labels in train_dataset:
        yield COMPAT_TRANSFORMS[shape_cls](pointcloud)


def get_class_objs(obj_dir, shape_cls, split="all"):
    """
    Get the list of objects for a given class/split.
    """
    compat_cls_code = int_to_hex(COMPAT_CLASSES[shape_cls])
    obj_files = os.listdir(obj_dir)
    obj_files = [os.path.join(obj_dir, f) for f in obj_files]
    obj_files = [
        f for f in obj_files if f.endswith(".obj") and compat_cls_code + "_" in f
    ]
    obj_files = sorted(obj_files)

    if split == "all":
        return obj_files

    # Open the split metadata
    pwd = os.path.dirname(os.path.realpath(__file__))
    split_dict = json.load(open(os.path.join(pwd, "CoMPaT", "split.json")))

    # Filter split meshes
    obj_files = [
        f for f in obj_files if f.split("/")[-1].split(".")[0] in split_dict[split]
    ]

    return obj_files
