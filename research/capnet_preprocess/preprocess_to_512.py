#!/usr/bin/env python
"""
CAP-net rendered data preprocessing for training.

Pipeline:
1) Load RGB + segmentation npz
2) Compute foreground bbox from (depth_segmentation OR semantic!=-2)
3) Crop with margin
4) Optional: overwrite background pixels (semantic==-2) in RGB to uniform gray
5) Convert to 512x512 with configurable strategy:
   - crop_resize: crop then resize to 512x512 (no padding, may distort aspect)
   - fit_pad_bg: keep aspect ratio, fit into canvas*fill_fraction, pad; padding uses bg color or gray
   - pad_to_canvas: paste crop into 512 canvas (may pad); if oversize, optionally downscale to fit

Masks are always resized with nearest neighbor (order=0) to preserve discrete labels (including negatives).
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from scipy import ndimage as ndi  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("This script requires scipy (already in requirements.txt).") from e


@dataclass(frozen=True)
class CropBox:
    y0: int
    x0: int
    y1: int  # exclusive
    x1: int  # exclusive

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    @property
    def w(self) -> int:
        return self.x1 - self.x0


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _load_rgb(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _save_rgb(path: str, rgb: np.ndarray) -> None:
    _ensure_dir(os.path.dirname(path))
    Image.fromarray(rgb).save(path)


def _load_npz(path: str) -> Dict[str, np.ndarray]:
    z = np.load(path)
    return {k: z[k] for k in z.files}


def _save_npz(path: str, arrays: Dict[str, np.ndarray]) -> None:
    _ensure_dir(os.path.dirname(path))
    np.savez_compressed(path, **arrays)


def _foreground_mask(seg: Dict[str, np.ndarray]) -> np.ndarray:
    dep = seg.get("depth_segmentation")
    sem = seg.get("semantic_segmentation")
    if dep is None and sem is None:
        raise KeyError("Need at least one of: depth_segmentation, semantic_segmentation")
    dep_fg = dep.astype(bool) if dep is not None else None
    sem_fg = (sem != -2) if sem is not None else None
    if dep_fg is None:
        return sem_fg.astype(bool)
    if sem_fg is None:
        return dep_fg
    return dep_fg | sem_fg.astype(bool)


def compute_bbox(fg: np.ndarray) -> Optional[CropBox]:
    ys, xs = np.where(fg)
    if ys.size == 0:
        return None
    return CropBox(int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1)


def expand_bbox(box: CropBox, h: int, w: int, margin: int) -> CropBox:
    return CropBox(
        y0=max(0, box.y0 - margin),
        x0=max(0, box.x0 - margin),
        y1=min(h, box.y1 + margin),
        x1=min(w, box.x1 + margin),
    )


def crop_arrays(
    rgb: np.ndarray, seg: Dict[str, np.ndarray], box: CropBox
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    rgb_c = rgb[box.y0 : box.y1, box.x0 : box.x1]
    out: Dict[str, np.ndarray] = {}
    for k, a in seg.items():
        if a.ndim != 2:
            raise ValueError(f"Expected 2D array for {k}, got shape={a.shape}")
        out[k] = a[box.y0 : box.y1, box.x0 : box.x1]
    return rgb_c, out


def apply_uniform_bg(rgb: np.ndarray, seg: Dict[str, np.ndarray], *, gray: int) -> np.ndarray:
    sem = seg.get("semantic_segmentation")
    if sem is None or sem.shape != rgb.shape[:2]:
        return rgb
    out = rgb.copy()
    m = sem == -2
    if m.any():
        out[m] = (gray, gray, gray)
    return out


def _resize_rgb(rgb: np.ndarray, new_hw: Tuple[int, int]) -> np.ndarray:
    nh, nw = new_hw
    pil = Image.fromarray(rgb, mode="RGB")
    pil2 = pil.resize((nw, nh), resample=Image.BILINEAR)
    return np.asarray(pil2, dtype=np.uint8)


def _resize_mask_nearest(a: np.ndarray, new_hw: Tuple[int, int]) -> np.ndarray:
    nh, nw = new_hw
    h, w = a.shape
    if (h, w) == (nh, nw):
        return a
    zoom_y = nh / float(h)
    zoom_x = nw / float(w)
    out = ndi.zoom(a, zoom=(zoom_y, zoom_x), order=0, mode="nearest", prefilter=False)
    out = out[:nh, :nw]
    if out.shape != (nh, nw):
        pad_h = nh - out.shape[0]
        pad_w = nw - out.shape[1]
        out = np.pad(out, ((0, max(0, pad_h)), (0, max(0, pad_w))), mode="edge")[:nh, :nw]
    return out.astype(a.dtype, copy=False)


def paste_to_canvas(
    rgb_c: np.ndarray,
    seg_c: Dict[str, np.ndarray],
    *,
    canvas: int,
    rgb_fill: Tuple[int, int, int],
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Tuple[int, int]]:
    ch, cw = rgb_c.shape[:2]
    if ch > canvas or cw > canvas:
        raise ValueError(f"Cropped region {ch}x{cw} does not fit into {canvas}x{canvas}")
    top = (canvas - ch) // 2
    left = (canvas - cw) // 2

    rgb_out = np.zeros((canvas, canvas, 3), dtype=np.uint8)
    rgb_out[:, :] = np.array(rgb_fill, dtype=np.uint8)[None, None, :]
    rgb_out[top : top + ch, left : left + cw] = rgb_c

    seg_out: Dict[str, np.ndarray] = {}
    for k, a in seg_c.items():
        if k in ("semantic_segmentation", "instance_segmentation"):
            out = np.full((canvas, canvas), np.int32(-2), dtype=np.int32)
        elif k == "depth_segmentation":
            out = np.zeros((canvas, canvas), dtype=bool)
        else:
            out = np.zeros((canvas, canvas), dtype=a.dtype)
        out[top : top + ch, left : left + cw] = a
        seg_out[k] = out

    return rgb_out, seg_out, (top, left)


def fit_into_canvas(
    rgb_c: np.ndarray,
    seg_c: Dict[str, np.ndarray],
    *,
    canvas: int,
    rgb_fill: Tuple[int, int, int],
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Tuple[int, int], float]:
    ch, cw = rgb_c.shape[:2]
    if ch <= canvas and cw <= canvas:
        rgb_o, seg_o, (top, left) = paste_to_canvas(rgb_c, seg_c, canvas=canvas, rgb_fill=rgb_fill)
        return rgb_o, seg_o, (top, left), 1.0
    scale = min(canvas / float(ch), canvas / float(cw))
    nh = max(1, int(round(ch * scale)))
    nw = max(1, int(round(cw * scale)))
    rgb_r = _resize_rgb(rgb_c, (nh, nw))
    seg_r = {k: _resize_mask_nearest(v, (nh, nw)) for k, v in seg_c.items()}
    rgb_o, seg_o, (top, left) = paste_to_canvas(rgb_r, seg_r, canvas=canvas, rgb_fill=rgb_fill)
    return rgb_o, seg_o, (top, left), float(scale)


def crop_resize_to_square(
    rgb_c: np.ndarray, seg_c: Dict[str, np.ndarray], *, size: int
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    rgb_r = _resize_rgb(rgb_c, (size, size))
    seg_r = {k: _resize_mask_nearest(v, (size, size)) for k, v in seg_c.items()}
    return rgb_r, seg_r


def fit_pad(
    rgb_c: np.ndarray,
    seg_c: Dict[str, np.ndarray],
    *,
    canvas: int,
    fill_fraction: float,
    rgb_fill: Tuple[int, int, int],
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Tuple[int, int], float]:
    if not (0.1 <= fill_fraction <= 1.0):
        raise ValueError("fill_fraction must be in [0.1, 1.0]")
    ch, cw = rgb_c.shape[:2]
    target = max(1, int(round(canvas * fill_fraction)))
    scale = min(target / float(ch), target / float(cw))
    nh = max(1, int(round(ch * scale)))
    nw = max(1, int(round(cw * scale)))
    rgb_r = _resize_rgb(rgb_c, (nh, nw))
    seg_r = {k: _resize_mask_nearest(v, (nh, nw)) for k, v in seg_c.items()}
    rgb_o, seg_o, (top, left) = paste_to_canvas(rgb_r, seg_r, canvas=canvas, rgb_fill=rgb_fill)
    return rgb_o, seg_o, (top, left), float(scale)


def _colorize_mask(a: np.ndarray) -> np.ndarray:
    def label_to_color(label: int) -> Tuple[int, int, int]:
        if label == -2:
            return (0, 0, 0)
        if label == -1:
            return (110, 110, 110)
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

    out = np.zeros((a.shape[0], a.shape[1], 3), dtype=np.uint8)
    for lab in np.unique(a):
        out[a == lab] = label_to_color(int(lab))
    return out


def _load_default_font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def _add_legend_panel(
    img_rgb: np.ndarray,
    labels: List[int],
    *,
    title: str,
    panel_width: int = 300,
    swatch: int = 22,
    pad: int = 14,
    row_gap: int = 10,
) -> np.ndarray:
    """
    Concatenate a right-side legend panel: color swatch + label id text.
    """
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError("img_rgb must be HxWx3 (RGB)")

    h = img_rgb.shape[0]
    panel = Image.new("RGB", (panel_width, h), (18, 18, 18))
    draw = ImageDraw.Draw(panel)
    font = _load_default_font()

    y = pad
    draw.text((pad, y), title, fill=(245, 245, 245), font=font)
    y += swatch + row_gap

    # stable ordering: background -> unlabeled -> non-negative ascending
    ordered: List[int] = []
    for x in (-2, -1):
        if x in labels:
            ordered.append(x)
    ordered.extend(sorted([x for x in labels if x >= 0]))

    def label_to_color(label: int) -> Tuple[int, int, int]:
        if label == -2:
            return (0, 0, 0)
        if label == -1:
            return (110, 110, 110)
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

    for lab in ordered:
        color = label_to_color(int(lab))
        draw.rectangle([pad, y, pad + swatch, y + swatch], fill=color, outline=(230, 230, 230))
        draw.text((pad + swatch + 12, y + 3), f"label {int(lab)}", fill=(245, 245, 245), font=font)
        y += swatch + row_gap
        if y > h - pad - swatch:
            draw.text((pad, h - pad - swatch), "...", fill=(245, 245, 245), font=font)
            break

    return np.concatenate([img_rgb, np.asarray(panel, dtype=np.uint8)], axis=1)


def _overlay_mask_boundaries(rgb: np.ndarray, mask: np.ndarray, *, alpha: float = 0.45) -> np.ndarray:
    color = _colorize_mask(mask)
    blended = (rgb.astype(np.float32) * (1.0 - alpha) + color.astype(np.float32) * alpha).clip(
        0, 255
    ).astype(np.uint8)
    fg = mask != -2
    er = ndi.binary_erosion(
        fg, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    )
    boundary = fg & (~er)
    blended[boundary] = (255, 255, 255)
    return blended


def _stem_from_path(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]


def process_one(
    *,
    rgb_path: str,
    npz_path: str,
    out_root: str,
    canvas: int,
    margin: int,
    output_mode: str,
    fill_fraction: float,
    on_oversize: str,
    bg_fill: str,
    bg_gray: int,
    debug_vis: bool,
) -> Dict[str, object]:
    stem = _stem_from_path(rgb_path)
    row: Dict[str, object] = {"stem": stem, "rgb": rgb_path, "npz": npz_path}
    rgb = _load_rgb(rgb_path)
    seg = _load_npz(npz_path)

    required = {"semantic_segmentation", "instance_segmentation", "depth_segmentation"}
    if not required.issubset(seg.keys()):
        row["status"] = "missing_required_keys"
        row["keys"] = sorted(list(seg.keys()))
        return row

    fg = _foreground_mask(seg)
    box0 = compute_bbox(fg)
    if box0 is None:
        row["status"] = "empty_foreground"
        return row

    box = expand_bbox(box0, h=fg.shape[0], w=fg.shape[1], margin=margin)
    rgb_c, seg_c = crop_arrays(rgb, seg, box)
    row["crop_box_yxyx"] = [box.y0, box.x0, box.y1, box.x1]
    row["crop_hw"] = [int(rgb_c.shape[0]), int(rgb_c.shape[1])]

    if bg_fill == "gray":
        rgb_c = apply_uniform_bg(rgb_c, seg_c, gray=bg_gray)
        rgb_fill = (bg_gray, bg_gray, bg_gray)
    else:
        rgb_fill = (255, 255, 255)

    try:
        if output_mode == "crop_resize":
            rgb_512, seg_512 = crop_resize_to_square(rgb_c, seg_c, size=canvas)
        elif output_mode == "fit_pad_bg":
            rgb_512, seg_512, (top, left), scale = fit_pad(
                rgb_c, seg_c, canvas=canvas, fill_fraction=fill_fraction, rgb_fill=rgb_fill
            )
            row["paste_top_left"] = [int(top), int(left)]
            row["resize_scale"] = float(scale)
        else:  # pad_to_canvas
            if on_oversize == "resize":
                rgb_512, seg_512, (top, left), scale = fit_into_canvas(
                    rgb_c, seg_c, canvas=canvas, rgb_fill=rgb_fill
                )
                row["paste_top_left"] = [int(top), int(left)]
                row["resize_scale"] = float(scale)
            else:
                rgb_512, seg_512, (top, left) = paste_to_canvas(
                    rgb_c, seg_c, canvas=canvas, rgb_fill=rgb_fill
                )
                row["paste_top_left"] = [int(top), int(left)]
    except ValueError as e:
        if on_oversize == "error":
            raise
        row["status"] = "skip_oversize"
        row["reason"] = str(e)
        return row

    out_rgb_dir = os.path.join(out_root, "rgb_512")
    out_seg_dir = os.path.join(out_root, "segmentation_512")
    out_dbg_dir = os.path.join(out_root, "debug_vis")
    _ensure_dir(out_rgb_dir)
    _ensure_dir(out_seg_dir)
    _ensure_dir(out_dbg_dir)

    out_rgb_path = os.path.join(out_rgb_dir, f"{stem}.png")
    out_npz_path = os.path.join(out_seg_dir, f"{stem}.npz")
    _save_rgb(out_rgb_path, rgb_512)
    _save_npz(out_npz_path, seg_512)
    row["out_rgb"] = out_rgb_path
    row["out_npz"] = out_npz_path
    row["status"] = "ok"

    if debug_vis:
        sem = seg_512["semantic_segmentation"]
        ins = seg_512["instance_segmentation"]

        _save_rgb(os.path.join(out_dbg_dir, f"{stem}_rgb.png"), rgb_512)

        sem_color = _colorize_mask(sem)
        sem_overlay = _overlay_mask_boundaries(rgb_512, sem, alpha=0.45)
        sem_labels = [int(x) for x in np.unique(sem).tolist()]
        _save_rgb(os.path.join(out_dbg_dir, f"{stem}_semantic_color.png"), sem_color)
        _save_rgb(os.path.join(out_dbg_dir, f"{stem}_semantic_overlay.png"), sem_overlay)
        _save_rgb(
            os.path.join(out_dbg_dir, f"{stem}_semantic_color_legend.png"),
            _add_legend_panel(sem_color, sem_labels, title="semantic_segmentation"),
        )
        _save_rgb(
            os.path.join(out_dbg_dir, f"{stem}_semantic_overlay_legend.png"),
            _add_legend_panel(sem_overlay, sem_labels, title="semantic_segmentation"),
        )

        ins_color = _colorize_mask(ins)
        ins_overlay = _overlay_mask_boundaries(rgb_512, ins, alpha=0.45)
        ins_labels = [int(x) for x in np.unique(ins).tolist()]
        _save_rgb(os.path.join(out_dbg_dir, f"{stem}_instance_color.png"), ins_color)
        _save_rgb(os.path.join(out_dbg_dir, f"{stem}_instance_overlay.png"), ins_overlay)
        _save_rgb(
            os.path.join(out_dbg_dir, f"{stem}_instance_color_legend.png"),
            _add_legend_panel(ins_color, ins_labels, title="instance_segmentation"),
        )
        _save_rgb(
            os.path.join(out_dbg_dir, f"{stem}_instance_overlay_legend.png"),
            _add_legend_panel(ins_overlay, ins_labels, title="instance_segmentation"),
        )

    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True, help="Output root directory.")

    # batch mode
    ap.add_argument("--input_root", default=None, help="CAP-net sample_data root (contains rgb/ and segmentation/).")

    # single-sample mode
    ap.add_argument("--rgb", default=None, help="Single RGB path (png/jpg). If set, runs single-sample mode.")
    ap.add_argument("--npz", default=None, help="Single segmentation npz path. Used with --rgb.")

    ap.add_argument("--canvas", type=int, default=512, help="Target square size (default 512).")
    ap.add_argument("--margin", type=int, default=30, help="Expand bbox by this many pixels.")
    ap.add_argument(
        "--output_mode",
        choices=["fit_pad_bg", "crop_resize", "pad_to_canvas"],
        default="fit_pad_bg",
        help="Recommended: fit_pad_bg (keep aspect, pad).",
    )
    ap.add_argument(
        "--fill_fraction",
        type=float,
        default=0.75,
        help="Only for fit_pad_bg: content fits within canvas*fill_fraction. Smaller => more background.",
    )
    ap.add_argument(
        "--bg_fill",
        choices=["none", "gray"],
        default="gray",
        help="If gray: overwrite semantic==-2 pixels in RGB and padding with uniform gray (recommended).",
    )
    ap.add_argument("--bg_gray", type=int, default=160, help="Gray value used when --bg_fill gray.")
    ap.add_argument(
        "--on_oversize",
        choices=["skip", "error", "resize"],
        default="resize",
        help="Only for pad_to_canvas: what to do if crop doesn't fit in canvas.",
    )
    ap.add_argument(
        "--debug_first_vis",
        action="store_true",
        help="In batch mode: save debug vis for the first successful sample.",
    )
    ap.add_argument(
        "--debug_vis",
        action="store_true",
        help="In single mode: save debug vis for this sample.",
    )
    args = ap.parse_args()

    if args.rgb is not None:
        if args.npz is None:
            raise ValueError("When using --rgb (single mode), you must also provide --npz.")
        row = process_one(
            rgb_path=args.rgb,
            npz_path=args.npz,
            out_root=args.out_root,
            canvas=int(args.canvas),
            margin=int(args.margin),
            output_mode=args.output_mode,
            fill_fraction=float(args.fill_fraction),
            on_oversize=args.on_oversize,
            bg_fill=args.bg_fill,
            bg_gray=int(args.bg_gray),
            debug_vis=bool(args.debug_vis),
        )
        report = {
            "mode": "single",
            "out_root": os.path.abspath(args.out_root),
            "canvas": int(args.canvas),
            "margin": int(args.margin),
            "output_mode": args.output_mode,
            "fill_fraction": float(args.fill_fraction),
            "bg_fill": args.bg_fill,
            "bg_gray": int(args.bg_gray),
            "row": row,
        }
        with open(os.path.join(args.out_root, "preprocess_report_single.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    if args.input_root is None:
        raise ValueError("Provide either --input_root (batch) or --rgb/--npz (single).")

    rgb_dir = os.path.join(args.input_root, "rgb")
    seg_dir = os.path.join(args.input_root, "segmentation")
    if not os.path.isdir(rgb_dir) or not os.path.isdir(seg_dir):
        raise FileNotFoundError(f"input_root must contain rgb/ and segmentation/: {args.input_root}")

    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    rows: List[Dict[str, object]] = []
    dbg_done = False

    for fn in rgb_files:
        stem = os.path.splitext(fn)[0]
        rgb_path = os.path.join(rgb_dir, fn)
        npz_path = os.path.join(seg_dir, f"{stem}.npz")
        if not os.path.exists(npz_path):
            rows.append({"stem": stem, "rgb": rgb_path, "npz": npz_path, "status": "missing_npz"})
            continue

        row = process_one(
            rgb_path=rgb_path,
            npz_path=npz_path,
            out_root=args.out_root,
            canvas=int(args.canvas),
            margin=int(args.margin),
            output_mode=args.output_mode,
            fill_fraction=float(args.fill_fraction),
            on_oversize=args.on_oversize,
            bg_fill=args.bg_fill,
            bg_gray=int(args.bg_gray),
            debug_vis=bool(args.debug_first_vis and (not dbg_done)),
        )
        rows.append(row)
        if args.debug_first_vis and (not dbg_done) and row.get("status") == "ok":
            dbg_done = True

    report = {
        "mode": "batch",
        "input_root": os.path.abspath(args.input_root),
        "out_root": os.path.abspath(args.out_root),
        "canvas": int(args.canvas),
        "margin": int(args.margin),
        "output_mode": args.output_mode,
        "fill_fraction": float(args.fill_fraction),
        "bg_fill": args.bg_fill,
        "bg_gray": int(args.bg_gray),
        "n_total": len(rgb_files),
        "n_ok": sum(1 for r in rows if r.get("status") == "ok"),
        "rows": rows,
    }
    with open(os.path.join(args.out_root, "preprocess_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: report[k] for k in report if k != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

