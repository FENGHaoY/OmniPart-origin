# -*- coding: utf-8 -*-
"""
对每张图的 SAM 候选区域提取 DINOv2 特征，写入 region_features/。

功能：
- 读取 pseudo_seg/indexes/image_index.json 与 proposals/<category>/<model_id>/view_xxx_sam_masks.json|.npz
- 对每张图：加载图像 → DINOv2 得到 patch 特征 → 按每个 proposal mask 做 masked mean pool → 每区域一向量
- 输出 region_features/<category>/<model_id>/view_xxx_region_features.npz（features [N,D]）+ .json（元信息）
- 小批量参数与 run_sam_proposals 一致：--limit_per_category、--limit_views_per_object、--debug_single_object、--dry_run
详见 research/docs/design_pseudo_seg_pipeline.md
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# DINOv2：与 OmniPart 一致用 HuggingFace transformers
try:
    from transformers import AutoModel
except ImportError:
    raise ImportError("请安装 transformers，用于加载 DINOv2")

DEFAULT_CATEGORIES = [
    "Dishwasher",
    "Microwave",
    "Oven",
    "Refrigerator",
    "StorageFurniture",
    "Table",
    "WashingMachine",
]

# DINOv2 ViT-L/14：max_size=518，patch_size=14 → 37x37 个 patch
DINOV2_MAX_SIZE = 518
DINOV2_PATCH_SIZE = 14
DINOV2_MODEL_NAME = "facebook/dinov2-large"


def resolve_default_paths() -> Dict[str, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    project_root = repo_root.parent / "project"
    processed_root = project_root / "processed_data"
    pseudo_seg_root = processed_root / "pseudo_seg"
    return {
        "pseudo_seg_root": pseudo_seg_root,
        "image_index": pseudo_seg_root / "indexes" / "image_index.json",
        "proposals_dir": pseudo_seg_root / "proposals",
        "region_features_dir": pseudo_seg_root / "region_features",
        "logs_dir": pseudo_seg_root / "logs",
        "stats_dir": pseudo_seg_root / "stats",
    }


def load_image_index(index_path: Path) -> Dict[str, Any]:
    if not index_path.is_file():
        raise FileNotFoundError(f"image_index 不存在: {index_path}")
    with index_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_image_rgb(image_path: Path) -> np.ndarray:
    try:
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        return np.array(img)
    except Exception:
        import cv2
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_proposals_for_view(proposals_dir: Path, view_id: int) -> Tuple[Optional[Dict], Optional[np.ndarray]]:
    """加载 view_xxx_sam_masks.json 与 view_xxx_sam_masks.npz，返回 (meta, masks_array)。"""
    prefix = f"view_{view_id:03d}_sam_masks"
    json_path = proposals_dir / f"{prefix}.json"
    npz_path = proposals_dir / f"{prefix}.npz"
    if not json_path.is_file() or not npz_path.is_file():
        return None, None
    with json_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    data = np.load(npz_path)
    masks_arr = data["masks"]  # (N, H, W) uint8
    return meta, masks_arr


def build_dinov2_encoder(device: str, model_name: str = DINOV2_MODEL_NAME) -> Tuple[Any, int, int, int]:
    """
    加载 DINOv2 并返回 (model, patch_h, patch_w, feature_dim)。
    last_hidden_state 形状为 [B, 1+H*W, D]，去掉 CLS 后 reshape 为 (H, W, D)。
    """
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
    model.requires_grad_(False)
    model.eval()
    model = model.to(device)
    patch_h = DINOV2_MAX_SIZE // DINOV2_PATCH_SIZE
    patch_w = patch_h
    feature_dim = getattr(model.config, "hidden_size", 1024)
    return model, patch_h, patch_w, feature_dim


def preprocess_image_for_dinov2(image_rgb: np.ndarray) -> Tuple[torch.Tensor, int, int]:
    """
    将 (H, W, 3) uint8 缩放到 max_size=518 且保持比例（居中 pad 到 518x518），
    并做 ImageNet 归一化。返回 (1, 3, 518, 518) tensor 与原始 H, W。
    """
    from PIL import Image
    h, w = image_rgb.shape[0], image_rgb.shape[1]
    scale = DINOV2_MAX_SIZE / max(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    img = Image.fromarray(image_rgb).resize((new_w, new_h), Image.BILINEAR)
    img = np.array(img)
    # Pad to 518x518
    pad_h = DINOV2_MAX_SIZE - img.shape[0]
    pad_w = DINOV2_MAX_SIZE - img.shape[1]
    top, left = pad_h // 2, pad_w // 2
    padded = np.pad(
        img,
        ((top, pad_h - top), (left, pad_w - left), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    # (518, 518, 3) -> normalize
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (padded.astype(np.float32) / 255.0 - mean) / std
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)  # (1, 3, 518, 518)
    return x, h, w


def get_patch_features(image_tensor: torch.Tensor, model: Any, device: str) -> torch.Tensor:
    """
    image_tensor: (1, 3, 518, 518), 已归一化。
    返回 patch 特征 (1, patch_h, patch_w, D)，不含 CLS。
    """
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        out = model(image_tensor)
    # last_hidden_state: [1, 1+37*37, D] 或 [1, 37*37, D]
    hidd = out.last_hidden_state
    B, L, D = hidd.shape
    patch_h = patch_w = DINOV2_MAX_SIZE // DINOV2_PATCH_SIZE
    if L == 1 + patch_h * patch_w:
        hidd = hidd[:, 1:, :]
    elif L != patch_h * patch_w:
        raise ValueError(f"DINOv2 输出长度 {L} 与预期 patch 数 {patch_h * patch_w} 不符")
    # (1, 1369, D) -> (1, 37, 37, D)
    hidd = hidd.reshape(1, patch_h, patch_w, D)
    return hidd


def pool_region_features(
    patch_features: torch.Tensor,
    masks_arr: np.ndarray,
    orig_h: int,
    orig_w: int,
    device: str,
) -> np.ndarray:
    """
    patch_features: (1, patch_h, patch_w, D)
    masks_arr: (N, H, W) uint8，H/W 为原图尺寸。
    将每个 mask 下采样到 (patch_h, patch_w)，在 patch 空间做 masked mean pool，得到 (N, D)。
    """
    ph, pw = patch_features.shape[1], patch_features.shape[2]
    N = masks_arr.shape[0]
    D = patch_features.shape[3]
    feats = patch_features[0]  # (ph, pw, D)
    results = []
    for i in range(N):
        mask = masks_arr[i].astype(np.float32)  # (H, W)
        # 下采样到 (ph, pw)
        mask_small = torch.from_numpy(mask).to(device).unsqueeze(0).unsqueeze(0)
        mask_small = F.interpolate(
            mask_small,
            size=(ph, pw),
            mode="nearest",
        ).squeeze(0).squeeze(0)  # (ph, pw)
        mask_small = (mask_small > 0.5).float()
        count = mask_small.sum()
        if count < 1e-5:
            results.append(torch.zeros(D, device=device, dtype=feats.dtype))
            continue
        # (ph, pw, D) * (ph, pw, 1) -> sum / count
        feats_i = feats.to(device)
        pooled = (feats_i * mask_small.unsqueeze(-1)).sum(dim=(0, 1)) / count
        results.append(pooled)
    out = torch.stack(results).cpu().numpy().astype(np.float32)
    return out


def extract_and_save_region_features(
    proposals_dir: Path,
    region_features_dir: Path,
    view_id: int,
    image_path: Path,
    model: Any,
    device: str,
    patch_h: int,
    patch_w: int,
    log: logging.Logger,
) -> bool:
    """对单视角加载图与 proposals，提特征并写入 region_features/<cat>/<mid>/view_xxx_region_features.npz|.json。"""
    image_rgb = load_image_rgb(image_path)
    meta, masks_arr = load_proposals_for_view(proposals_dir, view_id)
    if meta is None or masks_arr is None:
        log.warning("无 proposals: %s view_%03d", proposals_dir, view_id)
        return False
    N = masks_arr.shape[0]
    feature_dim = getattr(model.config, "hidden_size", 1024)
    if N == 0:
        log.debug("view_%03d 无 proposal，跳过", view_id)
        out_dir = region_features_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_dir / f"view_{view_id:03d}_region_features.npz",
            features=np.zeros((0, feature_dim), dtype=np.float32),
        )
        with (out_dir / f"view_{view_id:03d}_region_features.json").open("w", encoding="utf-8") as f:
            json.dump({"view_id": view_id, "n_regions": 0, "feature_dim": feature_dim}, f, indent=2)
        return True

    image_tensor, orig_h, orig_w = preprocess_image_for_dinov2(image_rgb)
    patch_features = get_patch_features(image_tensor, model, device)
    features = pool_region_features(
        patch_features,
        masks_arr,
        orig_h,
        orig_w,
        device,
    )
    out_dir = region_features_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"view_{view_id:03d}_region_features.npz", features=features)
    meta_out = {
        "view_id": view_id,
        "image_path": str(image_path),
        "image_size": [orig_h, orig_w],
        "n_regions": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
    }
    with (out_dir / f"view_{view_id:03d}_region_features.json").open("w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2, ensure_ascii=False)
    log.debug("已写入 view_%03d 特征: %s", view_id, features.shape)
    return True


def setup_logging(logs_dir: Path) -> logging.Logger:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "extract_region_features.log"
    log = logging.getLogger("extract_region_features")
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(ch)
    return log


def main() -> None:
    paths = resolve_default_paths()
    import argparse
    parser = argparse.ArgumentParser(description="对 proposals 每区域提 DINOv2 特征，写 region_features/")
    parser.add_argument("--image_index", type=str, default=str(paths["image_index"]))
    parser.add_argument("--proposals_dir", type=str, default=str(paths["proposals_dir"]))
    parser.add_argument("--region_features_dir", type=str, default=str(paths["region_features_dir"]))
    parser.add_argument("--categories", type=str, nargs="*", default=None)
    parser.add_argument("--limit_per_category", type=int, default=None)
    parser.add_argument("--limit_views_per_object", type=int, default=None)
    parser.add_argument("--debug_single_object", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    image_index_path = Path(args.image_index).expanduser().resolve()
    proposals_dir = Path(args.proposals_dir).expanduser().resolve()
    region_features_dir = Path(args.region_features_dir).expanduser().resolve()
    logs_dir = paths["logs_dir"]
    stats_dir = paths["stats_dir"]
    log = setup_logging(logs_dir)

    log.info("加载 image_index: %s", image_index_path)
    index_data = load_image_index(image_index_path)
    objects = index_data.get("objects", [])
    categories = args.categories or DEFAULT_CATEGORIES
    objects = [o for o in objects if o.get("category") in categories]

    if args.debug_single_object:
        objects = [o for o in objects if o.get("object_id") == args.debug_single_object]
        if not objects:
            log.error("未找到 object_id=%s", args.debug_single_object)
            sys.exit(1)
    elif args.limit_per_category is not None:
        from collections import defaultdict
        per_cat = defaultdict(int)
        filtered = []
        for o in objects:
            cat = o.get("category", "")
            if per_cat[cat] >= args.limit_per_category:
                continue
            filtered.append(o)
            per_cat[cat] += 1
        objects = filtered

    if args.limit_views_per_object is not None:
        for o in objects:
            o["views"] = o.get("views", [])[: args.limit_views_per_object]
            o["n_views"] = len(o.get("views", []))

    total_views = sum(o.get("n_views", 0) for o in objects)
    log.info("待处理: %d objects, %d views", len(objects), total_views)
    if args.dry_run:
        log.info("dry_run 结束")
        return

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    log.info("加载 DINOv2: %s, device=%s", DINOV2_MODEL_NAME, device)
    model, patch_h, patch_w, _ = build_dinov2_encoder(device)

    n_ok = 0
    n_fail = 0
    for i, obj in enumerate(objects):
        object_id = obj.get("object_id", "?")
        category = obj.get("category", "unknown")
        model_id = obj.get("model_id", "unknown")
        views = obj.get("views", [])
        prop_dir = proposals_dir / category / model_id
        feat_dir = region_features_dir / category / model_id
        log.info("[%d/%d] %s (%d views)", i + 1, len(objects), object_id, len(views))
        if not prop_dir.is_dir():
            log.warning("proposals 目录不存在: %s", prop_dir)
            n_fail += len(views)
            continue
        for v in views:
            view_id = v.get("view_id", 0)
            img_path = Path(v.get("image_path_abs", ""))
            if not img_path.is_file():
                log.warning("图像不存在: %s", img_path)
                n_fail += 1
                continue
            try:
                ok = extract_and_save_region_features(
                    prop_dir,
                    feat_dir,
                    view_id,
                    img_path,
                    model,
                    device,
                    patch_h,
                    patch_w,
                    log,
                )
                if ok:
                    n_ok += 1
                else:
                    n_fail += 1
            except Exception as e:
                log.exception("提取特征失败 %s view_%s: %s", object_id, view_id, e)
                n_fail += 1

    stats = {
        "n_success": n_ok,
        "n_fail": n_fail,
        "n_total_views": n_ok + n_fail,
    }
    stats_dir.mkdir(parents=True, exist_ok=True)
    with (stats_dir / "extract_region_features_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    log.info("结束: 成功 %d, 失败 %d", n_ok, n_fail)


if __name__ == "__main__":
    main()
