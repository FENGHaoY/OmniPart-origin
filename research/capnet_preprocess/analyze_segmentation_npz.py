#!/usr/bin/env python
import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from scipy import ndimage as ndi  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("This script requires scipy (already in requirements.txt).") from e


@dataclass(frozen=True)
class MaskStats:
    shape: Tuple[int, int]
    dtype: str
    unique: List[int]
    counts: Dict[int, int]
    foreground_pixels: int
    labeled_pixels: int
    unlabeled_pixels: int
    labeled_coverage: float
    components_per_label: Dict[int, int]


def _sorted_unique_int(a: np.ndarray) -> List[int]:
    return [int(x) for x in np.unique(a).tolist()]


def _counts(a: np.ndarray) -> Dict[int, int]:
    u, c = np.unique(a, return_counts=True)
    return {int(uu): int(cc) for uu, cc in zip(u.tolist(), c.tolist())}


def _connected_components_per_label(a: np.ndarray, labels: Iterable[int]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)  # 4-neighborhood
    for lab in labels:
        if lab < 0:
            continue
        m = (a == lab)
        if not m.any():
            out[int(lab)] = 0
            continue
        _, n = ndi.label(m, structure=structure)
        out[int(lab)] = int(n)
    return out


def summarize_int_mask(a: np.ndarray, *, background_value: int = -2, unlabeled_value: int = -1) -> MaskStats:
    if a.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape={a.shape}")

    counts = _counts(a)
    unique = _sorted_unique_int(a)
    fg = a != background_value
    n_fg = int(fg.sum())
    n_unl = int((a == unlabeled_value).sum())
    n_lab = int(((a >= 0) & fg).sum())
    cov = float(n_lab / max(n_fg, 1))
    comps = _connected_components_per_label(a, [u for u in unique if u >= 0])

    return MaskStats(
        shape=(int(a.shape[0]), int(a.shape[1])),
        dtype=str(a.dtype),
        unique=unique,
        counts=counts,
        foreground_pixels=n_fg,
        labeled_pixels=n_lab,
        unlabeled_pixels=n_unl,
        labeled_coverage=cov,
        components_per_label=comps,
    )


def label_to_color(label: int) -> Tuple[int, int, int]:
    # RGB colors
    if label == -2:
        return (0, 0, 0)  # background
    if label == -1:
        return (110, 110, 110)  # unlabeled foreground / ignore
    palette = [
        (255, 59, 48),
        (52, 199, 89),
        (0, 122, 255),
        (255, 149, 0),
        (175, 82, 222),
        (255, 45, 85),
        (90, 200, 250),
        (255, 214, 10),
        (48, 209, 88),
        (191, 90, 242),
    ]
    return palette[int(label) % len(palette)]


def colorize_mask(a: np.ndarray) -> np.ndarray:
    out = np.zeros((a.shape[0], a.shape[1], 3), dtype=np.uint8)
    for lab in np.unique(a):
        out[a == lab] = label_to_color(int(lab))
    return out


def _load_default_font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def add_legend_panel(
    img_rgb: np.ndarray,
    labels: List[int],
    *,
    title: str,
    panel_width: int = 320,
    swatch: int = 22,
    pad: int = 16,
    row_gap: int = 10,
) -> np.ndarray:
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError("img_rgb must be HxWx3 (RGB)")

    h = img_rgb.shape[0]
    panel = Image.new("RGB", (panel_width, h), (18, 18, 18))
    draw = ImageDraw.Draw(panel)
    font = _load_default_font()

    y = pad
    draw.text((pad, y), title, fill=(245, 245, 245), font=font)
    y += swatch + row_gap

    ordered: List[int] = []
    for x in (-2, -1):
        if x in labels:
            ordered.append(x)
    ordered.extend(sorted([x for x in labels if x >= 0]))

    for lab in ordered:
        color = label_to_color(int(lab))
        draw.rectangle([pad, y, pad + swatch, y + swatch], fill=color, outline=(230, 230, 230))
        draw.text((pad + swatch + 12, y + 3), f"label {int(lab)}", fill=(245, 245, 245), font=font)
        y += swatch + row_gap
        if y > h - pad - swatch:
            draw.text((pad, h - pad - swatch), "...", fill=(245, 245, 245), font=font)
            break

    return np.concatenate([img_rgb, np.asarray(panel, dtype=np.uint8)], axis=1)


def overlay_boundaries(rgb: np.ndarray, mask: np.ndarray, *, alpha: float = 0.45) -> np.ndarray:
    color = colorize_mask(mask)
    blended = (rgb.astype(np.float32) * (1.0 - alpha) + color.astype(np.float32) * alpha).clip(
        0, 255
    ).astype(np.uint8)

    fg = mask != -2
    er = ndi.binary_erosion(fg, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool))
    boundary = fg & (~er)
    blended[boundary] = (255, 255, 255)
    return blended


def load_rgb(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def save_rgb(path: str, rgb: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(rgb).save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb", required=True, help="Path to RGB image (png/jpg).")
    ap.add_argument("--npz", required=True, help="Path to segmentation .npz.")
    ap.add_argument("--out_dir", required=True, help="Directory to write visualizations and report.json.")
    ap.add_argument("--alpha", type=float, default=0.45, help="Overlay alpha for color mask.")
    args = ap.parse_args()

    rgb = load_rgb(args.rgb)
    z = np.load(args.npz)

    required = ["semantic_segmentation", "instance_segmentation", "depth_segmentation"]
    missing = [k for k in required if k not in z.files]
    if missing:
        raise KeyError(f"Missing keys in npz: {missing}. Present keys: {z.files}")

    sem = z["semantic_segmentation"]
    ins = z["instance_segmentation"]
    dep = z["depth_segmentation"]

    if sem.shape != ins.shape or sem.shape != dep.shape:
        raise ValueError(
            f"Shape mismatch: semantic={sem.shape}, instance={ins.shape}, depth={dep.shape}"
        )
    if sem.shape[:2] != rgb.shape[:2]:
        raise ValueError(f"RGB shape {rgb.shape[:2]} != mask shape {sem.shape}")

    sem_stats = summarize_int_mask(sem)
    ins_stats = summarize_int_mask(ins)
    dep_counts = {bool(k): int(v) for k, v in zip(*np.unique(dep, return_counts=True))}

    fg = dep.astype(bool)
    pairs: Dict[str, int] = {}
    for s in np.unique(sem[fg]):
        for i in np.unique(ins[fg]):
            c = int(((sem == s) & (ins == i) & fg).sum())
            if c:
                pairs[f"{int(s)}->{int(i)}"] = c

    report = {
        "rgb_path": os.path.abspath(args.rgb),
        "npz_path": os.path.abspath(args.npz),
        "keys": list(z.files),
        "semantic": sem_stats.__dict__,
        "instance": ins_stats.__dict__,
        "depth": {"shape": list(dep.shape), "dtype": str(dep.dtype), "counts": dep_counts},
        "semantic_to_instance_pairs_on_depth_true": pairs,
    }

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    sem_color = colorize_mask(sem)
    ins_color = colorize_mask(ins)
    dep_vis = (dep.astype(np.uint8) * 255).astype(np.uint8)
    dep_vis_rgb = np.stack([dep_vis, dep_vis, dep_vis], axis=-1)

    sem_overlay = overlay_boundaries(rgb, sem, alpha=args.alpha)
    ins_overlay = overlay_boundaries(rgb, ins, alpha=args.alpha)

    sem_labels = [int(x) for x in np.unique(sem).tolist()]
    ins_labels = [int(x) for x in np.unique(ins).tolist()]
    sem_color_legend = add_legend_panel(sem_color, sem_labels, title="semantic_segmentation")
    ins_color_legend = add_legend_panel(ins_color, ins_labels, title="instance_segmentation")

    save_rgb(os.path.join(out_dir, "rgb.png"), rgb)
    save_rgb(os.path.join(out_dir, "semantic_color.png"), sem_color)
    save_rgb(os.path.join(out_dir, "semantic_color_legend.png"), sem_color_legend)
    save_rgb(os.path.join(out_dir, "instance_color.png"), ins_color)
    save_rgb(os.path.join(out_dir, "instance_color_legend.png"), ins_color_legend)
    save_rgb(os.path.join(out_dir, "depth_segmentation.png"), dep_vis_rgb)
    save_rgb(os.path.join(out_dir, "semantic_overlay.png"), sem_overlay)
    save_rgb(os.path.join(out_dir, "instance_overlay.png"), ins_overlay)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

