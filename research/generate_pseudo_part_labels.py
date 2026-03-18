# -*- coding: utf-8 -*-
"""
结合 proposals、region features、object_meta 生成 part-level 伪标签与 meta。

功能：
- 读取 pseudo_seg/proposals/、region_features/ 与 objects_meta/<category>/<model_id>/object_meta.json
- 利用 object_meta 的 part 数量（part_order）与结构先验，对每张图的 SAM 区域做：
  筛选小区域、按 DINOv2 特征聚类为 n_parts 类、region-to-part 分配、生成 part mask
- 输出 pseudo_masks/<category>/<model_id>/view_xxx_pseudo_partseg.png（0=背景，1..K=part）、view_xxx_pseudo_meta.json
- 小批量参数与前述脚本一致
详见 research/docs/design_pseudo_seg_pipeline.md
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans

DEFAULT_CATEGORIES = [
    "Dishwasher",
    "Microwave",
    "Oven",
    "Refrigerator",
    "StorageFurniture",
    "Table",
    "WashingMachine",
]

# 面积占比小于此的 proposal 不参与聚类，归入背景或合并到相邻
MIN_REGION_AREA_RATIO = 0.001


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
        "objects_meta_root": processed_root / "objects_meta",
        "pseudo_masks_dir": pseudo_seg_root / "pseudo_masks",
        "logs_dir": pseudo_seg_root / "logs",
        "stats_dir": pseudo_seg_root / "stats",
    }


def load_image_index(index_path: Path) -> Dict[str, Any]:
    if not index_path.is_file():
        raise FileNotFoundError(f"image_index 不存在: {index_path}")
    with index_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_object_meta(objects_meta_root: Path, category: str, model_id: str) -> Optional[Dict[str, Any]]:
    path = objects_meta_root / category / model_id / "object_meta.json"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_proposals_for_view(proposals_dir: Path, view_id: int) -> Tuple[Optional[Dict], Optional[np.ndarray]]:
    prefix = f"view_{view_id:03d}_sam_masks"
    json_path = proposals_dir / f"{prefix}.json"
    npz_path = proposals_dir / f"{prefix}.npz"
    if not json_path.is_file() or not npz_path.is_file():
        return None, None
    with json_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    masks_arr = np.load(npz_path)["masks"]
    return meta, masks_arr


def load_region_features_for_view(region_features_dir: Path, view_id: int) -> Optional[np.ndarray]:
    npz_path = region_features_dir / f"view_{view_id:03d}_region_features.npz"
    if not npz_path.is_file():
        return None
    return np.load(npz_path)["features"]


def generate_pseudo_part_mask(
    masks_arr: np.ndarray,
    features: np.ndarray,
    n_parts: int,
    min_area_ratio: float,
    log: logging.Logger,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    对 (N, H, W) masks 与 (N, D) features，按 n_parts 聚类并生成 part mask。
    返回 (part_mask, region_to_part)，part_mask 为 (H,W) uint8，0=背景，1..n_parts=part；
    region_to_part 为 (N,) 每个 proposal 分配到的 part_id（0..n_parts-1），-1 表示丢弃/背景。
    """
    N, H, W = masks_arr.shape
    if N == 0 or n_parts <= 0:
        part_mask = np.zeros((H, W), dtype=np.uint8)
        return part_mask, np.array([], dtype=np.int32)

    total_pixels = H * W
    areas = masks_arr.sum(axis=(1, 2))
    # 筛选：面积过小的不参与聚类
    keep = areas >= (total_pixels * min_area_ratio)
    if keep.sum() == 0:
        part_mask = np.zeros((H, W), dtype=np.uint8)
        return part_mask, np.full(N, -1, dtype=np.int32)

    feats_keep = features[keep]
    n_keep = feats_keep.shape[0]
    if n_keep <= n_parts:
        # 区域数不足 n_parts：每个区域一类，其余 part 空缺
        k = n_keep
    else:
        k = n_parts

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_keep = kmeans.fit_predict(feats_keep)

    # 映射回全部 N 个 proposal：未 keep 的为 -1
    region_to_part = np.full(N, -1, dtype=np.int32)
    region_to_part[keep] = labels_keep

    # 构建 part_mask：按 proposal 顺序“先画先得”，每个像素取编号最小的覆盖该像素的 proposal，再映射到 part
    painted = np.zeros((H, W), dtype=np.int32)
    painted.fill(-1)
    for i in range(N):
        if region_to_part[i] < 0:
            continue
        mask = masks_arr[i].astype(bool)
        unpainted = mask & (painted < 0)
        painted[unpainted] = i

    # painted 中存的是 proposal 下标；再映射到 part_id（1-based，0 留给背景）
    part_mask = np.zeros((H, W), dtype=np.uint8)
    for i in range(N):
        if region_to_part[i] < 0:
            continue
        part_id_0based = region_to_part[i]
        part_mask[painted == i] = part_id_0based + 1

    return part_mask, region_to_part


def save_pseudo_view(
    pseudo_masks_dir: Path,
    view_id: int,
    part_mask: np.ndarray,
    object_meta: Dict[str, Any],
    n_parts: int,
    region_to_part: np.ndarray,
    log: logging.Logger,
) -> None:
    pseudo_masks_dir.mkdir(parents=True, exist_ok=True)
    # partseg PNG：单通道 uint8，0=背景，1..n_parts=part
    out_png = pseudo_masks_dir / f"view_{view_id:03d}_pseudo_partseg.png"
    try:
        from PIL import Image
        Image.fromarray(part_mask).save(out_png)
    except Exception:
        import cv2
        cv2.imwrite(str(out_png), part_mask)
    visible = sorted(set(np.unique(part_mask)) - {0})
    meta = {
        "view_id": view_id,
        "n_parts": n_parts,
        "part_order": object_meta.get("part_order", list(range(n_parts))),
        "visible_parts": visible,
        "mask_encoding": "0=background, 1..n_parts=part_id (1-based index into part_order)",
    }
    meta_path = pseudo_masks_dir / f"view_{view_id:03d}_pseudo_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    log.debug("已写入 %s: visible %s", out_png.name, visible)


def setup_logging(logs_dir: Path) -> logging.Logger:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "generate_pseudo_part_labels.log"
    log = logging.getLogger("generate_pseudo_part_labels")
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
    parser = argparse.ArgumentParser(description="结合 proposals、region features、object_meta 生成 pseudo part mask")
    parser.add_argument("--image_index", type=str, default=str(paths["image_index"]))
    parser.add_argument("--proposals_dir", type=str, default=str(paths["proposals_dir"]))
    parser.add_argument("--region_features_dir", type=str, default=str(paths["region_features_dir"]))
    parser.add_argument("--objects_meta_root", type=str, default=str(paths["objects_meta_root"]))
    parser.add_argument("--pseudo_masks_dir", type=str, default=str(paths["pseudo_masks_dir"]))
    parser.add_argument("--categories", type=str, nargs="*", default=None)
    parser.add_argument("--limit_per_category", type=int, default=None)
    parser.add_argument("--limit_views_per_object", type=int, default=None)
    parser.add_argument("--debug_single_object", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--min_region_area_ratio", type=float, default=MIN_REGION_AREA_RATIO)
    args = parser.parse_args()

    image_index_path = Path(args.image_index).expanduser().resolve()
    proposals_dir = Path(args.proposals_dir).expanduser().resolve()
    region_features_dir = Path(args.region_features_dir).expanduser().resolve()
    objects_meta_root = Path(args.objects_meta_root).expanduser().resolve()
    pseudo_masks_dir = Path(args.pseudo_masks_dir).expanduser().resolve()
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

    n_ok = 0
    n_fail = 0
    for i, obj in enumerate(objects):
        object_id = obj.get("object_id", "?")
        category = obj.get("category", "unknown")
        model_id = obj.get("model_id", "unknown")
        views = obj.get("views", [])
        object_meta = load_object_meta(objects_meta_root, category, model_id)
        if object_meta is None:
            log.warning("无 object_meta: %s/%s", category, model_id)
            n_fail += len(views)
            continue
        part_order = object_meta.get("part_order", [])
        n_parts = len(part_order)
        if n_parts == 0:
            log.warning("object_meta 无 part: %s", object_id)
            n_fail += len(views)
            continue

        prop_dir = proposals_dir / category / model_id
        feat_dir = region_features_dir / category / model_id
        out_dir = pseudo_masks_dir / category / model_id
        log.info("[%d/%d] %s n_parts=%d (%d views)", i + 1, len(objects), object_id, n_parts, len(views))
        if not prop_dir.is_dir():
            log.warning("proposals 目录不存在: %s", prop_dir)
            n_fail += len(views)
            continue
        for v in views:
            view_id = v.get("view_id", 0)
            meta, masks_arr = load_proposals_for_view(prop_dir, view_id)
            if meta is None or masks_arr is None:
                log.debug("无 proposals: %s view_%03d", object_id, view_id)
                n_fail += 1
                continue
            features = load_region_features_for_view(feat_dir, view_id)
            if features is None:
                log.warning("无 region features: %s view_%03d", object_id, view_id)
                n_fail += 1
                continue
            if features.shape[0] != masks_arr.shape[0]:
                log.warning("特征数与 mask 数不一致 view_%03d: %d vs %d", view_id, features.shape[0], masks_arr.shape[0])
                n_fail += 1
                continue
            try:
                part_mask, region_to_part = generate_pseudo_part_mask(
                    masks_arr,
                    features,
                    n_parts,
                    args.min_region_area_ratio,
                    log,
                )
                save_pseudo_view(out_dir, view_id, part_mask, object_meta, n_parts, region_to_part, log)
                n_ok += 1
            except Exception as e:
                log.exception("生成伪标签失败 %s view_%s: %s", object_id, view_id, e)
                n_fail += 1

    stats = {"n_success": n_ok, "n_fail": n_fail, "n_total_views": n_ok + n_fail}
    stats_dir.mkdir(parents=True, exist_ok=True)
    with (stats_dir / "generate_pseudo_part_labels_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    log.info("结束: 成功 %d, 失败 %d", n_ok, n_fail)


if __name__ == "__main__":
    main()
