"""
Defining all the ShapeNet class synsets as a dictionary.
"""

import numpy as np


# ShapeNet synsets
SHAPENET_CLASSES = {
    "airplane": "02691156",
    "bag": "02773838",
    "basket": "02801938",
    "bathtub": "02808440",
    "bed": "02818832",
    "bench": "02828884",
    "bicycle": "02834778",
    "birdhouse": "02843684",
    "bookshelf": "02871439",
    "bottle": "02876657",
    "bowl": "02880940",
    "bus": "02924116",
    "cabinet": "02933112",
    "camera": "02942699",
    "can": "02946921",
    "cap": "02954340",
    "car": "02958343",
    "chair": "03001627",
    "keyboard": "03085013",
    "dishwasher": "03207941",
    "display": "03211117",
    "earphone": "03261776",
    "faucet": "03325088",
    "file cabinet": "03337140",
    "guitar": "03467517",
    "jar": "03593526",
    "knife": "03624134",
    "lamp": "03636649",
    "laptop": "03642806",
    "loudspeaker": "03691459",
    "mailbox": "03710193",
    "microphone": "03759954",
    "motorbike": "03790512",
    "mug": "03797390",
    "piano": "03928116",
    "pillow": "03938244",
    "pistol": "03948459",
    "printer": "04004475",
    "remote": "04074963",
    "rifle": "04090263",
    "rocket": "04099429",
    "skateboard": "04225987",
    "sofa": "04256520",
    "stove": "04330267",
    "table": "04379243",
    "telephone": "04401088",
    "tower": "04460130",
    "train": "04468005",
    "trash bin": "02747177",
    "watercraft": "04530566",
    "washer": "04554684",
}

# 3DCoMPaT class codes
COMPAT_CLASSES = {
    "airplane": 0,
    "bag": 1,
    "basket": 2,
    "bed": 4,
    "bench": 5,
    "birdhouse": 7,
    "bookshelf": 29,
    "cabinet": 9,
    "car": 11,
    "chair": 12,
    "dishwasher": 16,
    "jar": 22,
    "lamp": 24,
    "skateboard": 32,
    "sofa": 33,
    "table": 37,
}


def int_to_hex(i):
    """
    Convert integer to hex string for 3DCoMPaT classes.
    """
    return str(hex(i)[2:].zfill(2))


def flip_yz(pc):
    """
    Flip Y and Z axis.
    """
    return pc[:, [0, 2, 1]]


def flip_zx(pc):
    """
    Flip Z and X axis.
    """
    return pc[:, [2, 1, 0]]


def flip_airplane(pc):
    """
    Rotate 180° around Y axis.
    """
    full_transform = np.array(
        [[-0.0, -1.0, -0.0], [0.0, 0.0, 1.0], [-1.0, -0.0, -0.0]]
    ).astype(np.float32)
    pc = np.dot(full_transform, pc.T).T
    return pc


def flip_bag(pc):
    """
    Rotate 90° around X axis.
    """
    full_transform = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    ).astype(np.float32)
    pc = np.dot(pc, full_transform)
    return pc


def flip_bench(pc):
    """
    Rotate 90° around X axis.
    Then flip front and back.
    """
    full_transform = np.array(
        [[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
    ).astype(np.float32)
    pc = np.dot(pc, full_transform)
    return pc


def flip_car(pc):
    """
    Rotate 90° around X axis.
    Then rotate 90° from left to right.
    """
    full_transform = np.array(
        [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    ).astype(np.float32)
    pc = np.dot(pc, full_transform)
    return pc


# Map compat classes to transformations
COMPAT_TRANSFORMS = {
    "airplane": flip_airplane,
    "bag": flip_bag,
    "basket": flip_bag,
    "bed": flip_bag,
    "bench": flip_bench,
    "birdhouse": flip_bench,
    "bookshelf": flip_bag,
    "cabinet": flip_bag,
    "car": flip_car,
    "chair": flip_yz,
    "dishwasher": flip_bench,
    "jar": flip_bag,
    "lamp": flip_bag,
    "skateboard": flip_car,
    "sofa": flip_bench,
    "table": flip_bag,
}
