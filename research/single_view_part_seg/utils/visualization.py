import os
from typing import Dict

import numpy as np
from PIL import Image


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def colorize_semantic(mask: np.ndarray) -> np.ndarray:
    """
    Unified semantic: 0 base,1 door,2 drawer,3 handle,4 knob,255 ignore
    """
    colors = {
        0: (160, 160, 160),  # base
        1: (255, 99, 71),    # door
        2: (30, 144, 255),   # drawer
        3: (60, 179, 113),   # handle
        4: (218, 165, 32),   # knob
        255: (0, 0, 0),      # ignore
    }
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for k, c in colors.items():
        out[mask == k] = c
    return out


def colorize_instance(mask: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    ids = np.unique(mask)
    for idx in ids:
        if idx == ignore_index:
            out[mask == idx] = (0, 0, 0)
            continue
        v = int(idx)
        # Deterministic pseudo-random palette
        color = ((37 * v + 53) % 255, (17 * v + 131) % 255, (97 * v + 29) % 255)
        out[mask == idx] = color
    return out


def save_image(path: str, rgb: np.ndarray) -> None:
    _ensure_dir(os.path.dirname(path))
    Image.fromarray(rgb.astype(np.uint8)).save(path)


def save_infer_visuals(
    out_dir: str,
    stem: str,
    image: np.ndarray,
    gt_sem: np.ndarray,
    pred_sem: np.ndarray,
    gt_ins: np.ndarray,
    pred_ins: np.ndarray,
) -> Dict[str, str]:
    _ensure_dir(out_dir)
    paths = {
        "image": os.path.join(out_dir, f"{stem}_image.png"),
        "gt_sem": os.path.join(out_dir, f"{stem}_gt_semantic.png"),
        "pred_sem": os.path.join(out_dir, f"{stem}_pred_semantic.png"),
        "gt_ins": os.path.join(out_dir, f"{stem}_gt_instance.png"),
        "pred_ins": os.path.join(out_dir, f"{stem}_pred_instance.png"),
    }
    save_image(paths["image"], image)
    save_image(paths["gt_sem"], colorize_semantic(gt_sem))
    save_image(paths["pred_sem"], colorize_semantic(pred_sem))
    save_image(paths["gt_ins"], colorize_instance(gt_ins))
    save_image(paths["pred_ins"], colorize_instance(pred_ins))
    return paths

