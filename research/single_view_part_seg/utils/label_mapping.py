from typing import Dict

import numpy as np
import torch

# Unified training label space (SINGAPO-aligned):
# 0 base, 1 door, 2 drawer, 3 handle, 4 knob, 255 ignore
IGNORE_INDEX = 255
NUM_CLASSES = 5

# CAPNet semantic -> unified semantic mapping
CAPNET_TO_UNIFIED: Dict[int, int] = {
    -2: 255,  # background -> ignore
    -1: 0,    # base
    0: 3,     # line_fixed_handle -> handle
    1: 3,     # round_fixed_handle -> handle
    2: 255,   # slider_button -> ignore
    3: 1,     # hinge_door -> door
    4: 2,     # slider_drawer -> drawer
    5: 255,   # slider_lid -> ignore
    6: 255,   # hinge_lid -> ignore
    7: 4,     # hinge_knob -> knob
    8: 255,   # hinge_handle -> ignore
}

UNIFIED_LABEL_NAMES = {
    0: "base",
    1: "door",
    2: "drawer",
    3: "handle",
    4: "knob",
    255: "ignore",
}


def map_capnet_semantic_to_unified(semantic: np.ndarray) -> np.ndarray:
    """
    Args:
        semantic: int array [H, W], CAPNet semantic ids.
    Returns:
        mapped: int64 array [H, W], values in {0,1,2,3,4,255}.
    """
    if semantic.ndim != 2:
        raise ValueError(f"semantic must be [H,W], got {semantic.shape}")
    out = np.full(semantic.shape, IGNORE_INDEX, dtype=np.int64)
    for src, dst in CAPNET_TO_UNIFIED.items():
        out[semantic == src] = dst
    return out


def filter_instance_with_valid_mask(instance: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """
    Keep instance ids only on valid semantic pixels; others become 255 (ignore).
    """
    if instance.shape != valid_mask.shape:
        raise ValueError(f"shape mismatch: instance {instance.shape} vs valid_mask {valid_mask.shape}")
    out = instance.astype(np.int64, copy=True)
    out[~valid_mask] = IGNORE_INDEX
    return out


def to_torch_bool(mask: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(mask.astype(np.bool_))

