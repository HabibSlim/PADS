"""
Defining point-based neural encoders.
"""

from models.points.pointbert import pointbert_g512_d12
from datasets.metadata import N_COMPAT_CLASSES, N_COMPAT_FINE_PARTS


def pointbert_g512_d12_compat():
    model = pointbert_g512_d12(
        num_classes=N_COMPAT_CLASSES, num_parts=N_COMPAT_FINE_PARTS
    )
    return model
