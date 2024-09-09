"""
Defining all the ShapeNet/3DCoMPaT class codes as dictionaries.
Also define class transformations to ensure alignment between ShapeNet and 3DCoMPaT classes.
"""

import torch


SHAPENET_CLASSES = {
    "02691156": 0,
    "02747177": 1,
    "02773838": 2,
    "02801938": 3,
    "02808440": 4,
    "02818832": 5,
    "02828884": 6,
    "02843684": 7,
    "02871439": 8,
    "02876657": 9,
    "02880940": 10,
    "02924116": 11,
    "02933112": 12,
    "02942699": 13,
    "02946921": 14,
    "02954340": 15,
    "02958343": 16,
    "02992529": 17,
    "03001627": 18,
    "03046257": 19,
    "03085013": 20,
    "03207941": 21,
    "03211117": 22,
    "03261776": 23,
    "03325088": 24,
    "03337140": 25,
    "03467517": 26,
    "03513137": 27,
    "03593526": 28,
    "03624134": 29,
    "03636649": 30,
    "03642806": 31,
    "03691459": 32,
    "03710193": 33,
    "03759954": 34,
    "03761084": 35,
    "03790512": 36,
    "03797390": 37,
    "03928116": 38,
    "03938244": 39,
    "03948459": 40,
    "03991062": 41,
    "04004475": 42,
    "04074963": 43,
    "04090263": 44,
    "04099429": 45,
    "04225987": 46,
    "04256520": 47,
    "04330267": 48,
    "04379243": 49,
    "04401088": 50,
    "04460130": 51,
    "04468005": 52,
    "04530566": 53,
    "04554684": 54,
}


# ShapeNet synsets
SHAPENET_NAME_TO_SYNSET = {
    "airplane": "02691156",
    "bag": "02773838",
    "basket": "02801938",
    "bathtub": "02808440",
    "bed": "02818832",
    "bench": "02828884",
    "bicycle": "02834778",
    "bird_house": "02843684",
    "boat": "04530566",
    "bookshelf": "02871439",
    "bowl": "02880940",
    "bus": "02924116",
    "cabinet": "02933112",
    "camera": "02942699",
    "can": "02946921",
    "cap": "02954340",
    "car": "02958343",
    "chair": "03001627",
    "dresser": "02933112",  # EXTRA
    "love_seat": "03001627",
    "keyboard": "03085013",
    "dishwasher": "03207941",
    "display": "03211117",
    "earphone": "03261776",
    "faucet": "03325088",
    "file cabinet": "03337140",
    "guitar": "03467517",
    "jar": "03593526",
    "jug": "03593526",
    "knife": "03624134",
    "lamp": "03636649",
    "laptop": "03642806",
    "loudspeaker": "03691459",
    "mailbox": "03710193",
    "microphone": "03759954",
    "motorbike": "03790512",
    "mug": "03797390",
    "ottoman": "02828884",
    "piano": "03928116",
    "pillow": "03938244",
    "pistol": "03948459",
    "planter": "03593526",
    "printer": "04004475",
    "remote": "04074963",
    "rifle": "04090263",
    "rocket": "04099429",
    "shelf": "02871439",
    "skateboard": "04225987",
    "sofa": "04256520",
    "sports_table": "04379243",
    "stove": "04330267",
    "stool": "03001627",
    "table": "04379243",
    "telephone": "04401088",
    "tower": "04460130",
    "train": "04468005",
    "trashcan": "02747177",  # orig: trash bin
    "vase": "03593526",
    "watercraft": "04530566",
    "washer": "04554684",
}


SHAPENET_NAME_TO_SYNSET_INDEX = {
    name: SHAPENET_CLASSES[SHAPENET_NAME_TO_SYNSET[name]]
    for name in SHAPENET_NAME_TO_SYNSET
    if SHAPENET_NAME_TO_SYNSET[name] in SHAPENET_CLASSES
}

COMPAT_COARSE_PARTS = 43
COMPAT_FINE_PARTS = 275

# 3DCoMPaT class codes
COMPAT_CLASSES = [
    "airplane",
    "bag",
    "basket",
    "bbq_grill",
    "bed",
    "bench",
    "bicycle",
    "bird_house",
    "boat",
    "cabinet",
    "candle_holder",
    "car",
    "chair",
    "clock",
    "coat_rack",
    "curtain",
    "dishwasher",
    "dresser",
    "fan",
    "faucet",
    "trashcan",  # orig: garbage_bin
    "gazebo",
    "jug",
    "ladder",
    "lamp",
    "love_seat",
    "ottoman",
    "parasol",
    "planter",
    "shelf",
    "shower",
    "sinks",
    "skateboard",
    "sofa",
    "sports_table",
    "stool",
    "sun_lounger",
    "table",
    "toilet",
    "tray",
    "trolley",
    "vase",
]
COMPAT_CLASSES = {c: i for i, c in enumerate(COMPAT_CLASSES)}
COMPAT_MATCHED_CLASSES = [
    "airplane",
    "bag",
    "basket",
    "bed",
    "bench",
    "bird_house",
    "boat",
    "cabinet",
    "car",
    "chair",
    "dishwasher",
    "dresser",
    "faucet",
    "jug",
    "lamp",
    "love_seat",
    "ottoman",
    "planter",
    "shelf",
    "skateboard",
    "sofa",
    "sports_table",
    "stool",
    "table",
    "trashcan",
    "vase",
]
COMPAT_NO_MATCH = ["bbq_grill", "bicycle", "candle_holder"]


def int_to_hex(i):
    """
    Convert integer to hex string for 3DCoMPaT classes.
    """
    return str(hex(i)[2:].zfill(2))


def class_to_hex(class_name):
    """
    Convert class name to hex.
    """
    return int_to_hex(COMPAT_CLASSES[class_name])


def apply_transform(pc, transform_mat):
    """
    Apply a transformation matrix to a point cloud.
    """
    full_transform = torch.tensor(
        transform_mat,
        dtype=torch.float32,
        device=pc.device,
    )
    return torch.matmul(pc.squeeze(), full_transform).unsqueeze(0)


flip_front_to_back = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]
flip_front_to_right = [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
flip_front_to_left = [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]

# Map compat classes to transformations
COMPAT_TRANSFORMS = {
    "airplane": flip_front_to_right,
    "bag": flip_front_to_back,
    "basket": flip_front_to_left,
    "bed": flip_front_to_back,
    "bench": flip_front_to_back,
    "bird_house": flip_front_to_back,
    "boat": flip_front_to_right,
    "cabinet": flip_front_to_left,
    "dresser": flip_front_to_right,
    "car": flip_front_to_right,
    "chair": flip_front_to_back,
    "dishwasher": flip_front_to_back,
    "faucet": flip_front_to_back,
    "jug": flip_front_to_right,
    "lamp": flip_front_to_right,
    "love_seat": flip_front_to_back,
    "ottoman": flip_front_to_back,
    "planter": flip_front_to_right,
    "shelf": flip_front_to_right,
    "skateboard": flip_front_to_right,
    "sofa": flip_front_to_back,
    "sports_table": flip_front_to_right,
    "stool": flip_front_to_back,
    "table": flip_front_to_right,
    "trashcan": flip_front_to_right,
    "vase": flip_front_to_right,
}

SHAPENET_TRANSFORMS = {
    "bag": flip_front_to_back,
    "basket": flip_front_to_left,
    "cabinet": flip_front_to_left,
    "dresser": flip_front_to_left,
    "shelf": flip_front_to_left,
    "table": flip_front_to_right,
    "trashcan": flip_front_to_left,
}


def get_shapenet_transform(class_name):
    """
    Get the transformation function for a class.
    """
    if class_name not in SHAPENET_TRANSFORMS:
        return lambda x: x
    return lambda x: apply_transform(x, SHAPENET_TRANSFORMS[class_name])


def get_compat_transform(class_name):
    """
    Get the transformation function for a class.
    """
    if class_name not in COMPAT_TRANSFORMS:
        return lambda x: x
    return lambda x: apply_transform(x, COMPAT_TRANSFORMS[class_name])
