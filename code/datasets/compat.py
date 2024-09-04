"""
Dataset iterators for 3DCoMPaT and ShapeNet.
"""

import numpy as np
import zipfile
from datasets.CoMPaT import compat3D
from datasets.metadata import (
    SHAPENET_CLASSES,
    COMPAT_CLASSES,
    COMPAT_TRANSFORMS,
)

ZIP_SRC = "/ibex/user/slimhy/surfaces.zip"


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


def compat_iterator(meta_dir, zip_path, shape_cls, num_points):
    """
    3DCoMPaT iterator.
    """
    train_dataset = compat3D.ShapeLoader(
        zip_path=zip_path,
        meta_dir=meta_dir,
        split="train",
        n_points=num_points,
        shuffle=True,
        seed=0,
        filter_class=COMPAT_CLASSES[shape_cls],
    )

    for shape_id, shape_label, pointcloud, point_part_labels in train_dataset:
        yield COMPAT_TRANSFORMS[shape_cls](pointcloud)
