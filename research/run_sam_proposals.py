# -*- coding: utf-8 -*-
"""
对 image_index 中每张图运行 SAM，生成候选区域并保存为 proposals。

功能：
- 读取 pseudo_seg/indexes/image_index.json
- 使用 SAM (build_sam + SamAutomaticMaskGenerator) 对每张图做 automatic mask generation
- 每张图输出：view_xxx_sam_masks.json（元信息）、view_xxx_sam_masks.npz（mask 数组，便于后续特征与伪标签）
- 支持小批量：--limit_per_category、--limit_views_per_object、--debug_single_object、--dry_run
- 默认 SAM 权重：../OmniPart/ckpt/sam_vit_h_4b8939.pth
详见 research/docs/design_pseudo_seg_pipeline.md
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

# 与 OmniPart 主工程一致：使用 build_sam + SamAutomaticMaskGenerator
try:
    from segment_anything import SamAutomaticMaskGenerator, build_sam
except ImportError:
    raise ImportError("请安装 segment_anything，与 OmniPart 主工程一致")

import torch

# 默认 7 类
DEFAULT_CATEGORIES = [
    "Dishwasher",
    "Microwave",
    "Oven",
    "Refrigerator",
    "StorageFurniture",
    "Table",
    "WashingMachine",
]


def resolve_default_paths() -> Dict[str, Path]:
    """路径：pseudo_seg 根、image_index、proposals 输出、日志。"""
    repo_root = Path(__file__).resolve().parents[1]
    project_root = repo_root.parent / "project"
    processed_root = project_root / "processed_data"
    pseudo_seg_root = processed_root / "pseudo_seg"
    # 默认 SAM 权重：与 OmniPart 主工程同目录下的 ckpt
    sam_ckpt = repo_root.parent / "OmniPart" / "ckpt" / "sam_vit_h_4b8939.pth"
    return {
        "pseudo_seg_root": pseudo_seg_root,
        "image_index": pseudo_seg_root / "indexes" / "image_index.json",
        "proposals_dir": pseudo_seg_root / "proposals",
        "logs_dir": pseudo_seg_root / "logs",
        "stats_dir": pseudo_seg_root / "stats",
        "sam_checkpoint": sam_ckpt,
    }


def load_image_index(index_path: Path) -> Dict[str, Any]:
    """加载 image_index.json，返回含 meta 与 objects 的字典。"""
    if not index_path.is_file():
        raise FileNotFoundError(f"image_index 不存在: {index_path}")
    with index_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_image_rgb(image_path: Path) -> np.ndarray:
    """加载图像为 RGB numpy (H, W, 3) uint8，供 SAM 使用。"""
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


def run_sam_on_image(
    image_rgb: np.ndarray,
    mask_generator: Any,
    log: logging.Logger,
) -> List[Dict[str, Any]]:
    """
    对单张 RGB 图运行 SAM automatic mask generation。
    返回 SAM 原始输出：list of dict，每项含 segmentation, area, bbox, predicted_iou, stability_score 等。
    """
    try:
        masks = mask_generator.generate(image_rgb)
    except Exception as e:
        log.warning("SAM generate 失败: %s", e)
        raise
    return masks


def _draw_sam_vis(image_rgb: np.ndarray, masks: List[Dict[str, Any]], alpha: float = 0.5) -> np.ndarray:
    """
    将 SAM 的多个 mask 画成彩色叠加图，与原图混合便于查看。

    - 背景（未被任何 proposal 覆盖的像素）：不单独上色，保持原图 + 半透明混合后的效果
      （即 overlay 初始为 0，未覆盖处最终是 原图*(1-alpha)，不是“编号 0”的掩码）。
    - 掩码编号：SAM 只输出前景区域，没有“背景掩码”。proposal 下标 0 表示第 1 个区域，
      1 表示第 2 个区域，以此类推；背景 = 不属于任何 proposal 的像素。
    - 每个 proposal 的颜色由固定 palette 按下标 i 生成（便于复现）。
    - 重叠处采用「先画先得」：只对尚未被更小编号覆盖的像素上色，这样编号 0、1、2… 在图上
      各自保持不同颜色，重叠区域显示编号较小的颜色。
    - 若已安装 cv2，会在每个区域质心处绘制掩码编号（白字黑边），便于可视化对照。
    返回 (H, W, 3) uint8。
    """
    H, W = image_rgb.shape[0], image_rgb.shape[1]
    overlay = np.zeros((H, W, 3), dtype=np.float64)  # 未覆盖处为 0，混合后即原图
    painted = np.zeros((H, W), dtype=bool)  # 已被更小编号上色的像素不再覆盖
    for i, m in enumerate(masks):  # i=0 是第 1 个 proposal，不是背景
        seg = m.get("segmentation")
        if seg is None or seg.shape[0] != H or seg.shape[1] != W:
            continue
        # 固定 palette：每个 proposal 不同颜色，便于区分
        hue = (i * 137) % 360 / 360.0
        r = 0.5 + 0.5 * np.sin(hue * 6.28)
        g = 0.5 + 0.5 * np.sin(hue * 6.28 + 2.09)
        b = 0.5 + 0.5 * np.sin(hue * 6.28 + 4.18)
        mask = seg.astype(bool) & (~painted)  # 只对尚未被上色的像素上色
        overlay[mask, 0] = r
        overlay[mask, 1] = g
        overlay[mask, 2] = b
        painted = painted | seg.astype(bool)  # 本 mask 覆盖的像素视为已上色
    vis = (image_rgb.astype(np.float64) / 255.0 * (1 - alpha) + np.clip(overlay, 0, 1) * alpha) * 255
    vis = np.clip(vis, 0, 255).astype(np.uint8)

    # 在每个 proposal 上标出掩码编号（质心处），便于可视化对照
    if cv2 is not None and masks:
        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        for i, m in enumerate(masks):
            seg = m.get("segmentation")
            if seg is None or seg.shape[0] != H or seg.shape[1] != W:
                continue
            ys, xs = np.where(seg)
            if len(ys) == 0 or len(xs) == 0:
                continue
            cy, cx = int(np.mean(ys)), int(np.mean(xs))
            label = str(i)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.4, min(1.2, (H + W) / 1200.0))
            thickness = max(1, int(round((H + W) / 800)))
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
            # 描边：先画黑色轮廓，再画白色字
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    cv2.putText(vis_bgr, label, (cx - tw // 2 + dx, cy + th // 2 + dy), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
            cv2.putText(vis_bgr, label, (cx - tw // 2, cy + th // 2), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        vis = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
    return vis


def save_proposals_for_view(
    out_dir: Path,
    view_id: int,
    image_path: str,
    image_size: List[int],
    masks: List[Dict[str, Any]],
    log: logging.Logger,
    image_rgb: Optional[np.ndarray] = None,
) -> None:
    """
    将单视角的 SAM 结果写入 view_xxx_sam_masks.json、view_xxx_sam_masks.npz，
    以及 view_xxx_sam_vis.png（彩色分割叠加图，便于肉眼检查）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"view_{view_id:03d}_sam_masks"
    json_path = out_dir / f"{prefix}.json"
    npz_path = out_dir / f"{prefix}.npz"
    vis_path = out_dir / f"view_{view_id:03d}_sam_vis.png"

    # 元信息（不把 segmentation 塞进 JSON，太大）
    meta = {
        "image_path": image_path,
        "image_size": image_size,
        "n_masks": len(masks),
        "proposals": [],
    }
    for i, m in enumerate(masks):
        meta["proposals"].append({
            "index": i,
            "area": int(m.get("area", 0)),
            "bbox": [int(x) for x in m.get("bbox", [0, 0, 0, 0])],
            "predicted_iou": float(m.get("predicted_iou", 0.0)),
            "stability_score": float(m.get("stability_score", 0.0)),
        })
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # mask 数组 (N, H, W) uint8：每个 proposal 一个 0/1 二值图，无单独“背景通道”
    # 背景 = 所有 masks[i] 在该像素均为 0；proposal 编号 0 表示第 1 个区域
    if masks:
        H, W = masks[0]["segmentation"].shape
        arr = np.zeros((len(masks), H, W), dtype=np.uint8)
        for i, m in enumerate(masks):
            arr[i] = m["segmentation"].astype(np.uint8)
        np.savez_compressed(npz_path, masks=arr)
        # 生成分割可视化图：彩色叠加在原图上
        if image_rgb is not None and image_rgb.shape[0] == H and image_rgb.shape[1] == W:
            vis = _draw_sam_vis(image_rgb, masks, alpha=0.5)
            try:
                from PIL import Image
                Image.fromarray(vis).save(vis_path)
            except Exception:
                import cv2
                cv2.imwrite(str(vis_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            log.debug("已写入 %s", vis_path.name)
    else:
        np.savez_compressed(npz_path, masks=np.zeros((0, image_size[0], image_size[1]), dtype=np.uint8))
    log.debug("已写入 %s %s", json_path.name, npz_path.name)


def setup_logging(logs_dir: Path) -> logging.Logger:
    """日志：文件 + 控制台。"""
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "run_sam_proposals.log"
    logger = logging.getLogger("run_sam_proposals")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def main() -> None:
    paths = resolve_default_paths()
    import argparse
    parser = argparse.ArgumentParser(description="对 image_index 中每张图运行 SAM，保存 proposals")
    parser.add_argument("--image_index", type=str, default=str(paths["image_index"]))
    parser.add_argument("--proposals_dir", type=str, default=str(paths["proposals_dir"]))
    parser.add_argument("--sam_checkpoint", type=str, default=str(paths["sam_checkpoint"]), help="SAM 权重路径，默认 ../OmniPart/ckpt/sam_vit_h_4b8939.pth")
    parser.add_argument("--categories", type=str, nargs="*", default=None)
    parser.add_argument("--limit_per_category", type=int, default=None)
    parser.add_argument("--limit_views_per_object", type=int, default=None)
    parser.add_argument("--debug_single_object", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu，默认自动")
    args = parser.parse_args()

    image_index_path = Path(args.image_index).expanduser().resolve()
    proposals_dir = Path(args.proposals_dir).expanduser().resolve()
    sam_ckpt = Path(args.sam_checkpoint).expanduser().resolve()
    logs_dir = paths["logs_dir"]
    stats_dir = paths["stats_dir"]
    log = setup_logging(logs_dir)

    if not sam_ckpt.is_file():
        log.error("SAM 权重不存在: %s", sam_ckpt)
        sys.exit(1)

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

    # 限制每 object 的 view 数
    if args.limit_views_per_object is not None:
        for o in objects:
            o["views"] = o.get("views", [])[: args.limit_views_per_object]
            o["n_views"] = len(o["views"])

    total_views = sum(o.get("n_views", 0) for o in objects)
    log.info("待处理: %d objects, %d views", len(objects), total_views)
    if args.dry_run:
        log.info("dry_run 结束")
        return

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    log.info("加载 SAM，checkpoint=%s, device=%s", sam_ckpt, device)
    sam_model = build_sam(checkpoint=str(sam_ckpt)).to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam_model)

    n_ok = 0
    n_fail = 0
    proposal_counts: List[int] = []
    errors: List[Dict[str, str]] = []

    for i, obj in enumerate(objects):
        object_id = obj.get("object_id", "?")
        category = obj.get("category", "unknown")
        model_id = obj.get("model_id", "unknown")
        views = obj.get("views", [])
        out_obj_dir = proposals_dir / category / model_id
        log.info("[%d/%d] %s (%d views)", i + 1, len(objects), object_id, len(views))
        for v in views:
            view_id = v.get("view_id", 0)
            img_path = Path(v.get("image_path_abs", ""))
            if not img_path.is_file():
                log.warning("图像不存在: %s", img_path)
                n_fail += 1
                errors.append({"object_id": object_id, "view_id": view_id, "error": "image not found"})
                continue
            try:
                image = load_image_rgb(img_path)
                masks = run_sam_on_image(image, mask_generator, log)
                save_proposals_for_view(
                    out_obj_dir,
                    view_id,
                    str(img_path),
                    [image.shape[0], image.shape[1]],
                    masks,
                    log,
                    image_rgb=image,
                )
                n_ok += 1
                proposal_counts.append(len(masks))
            except Exception as e:
                log.exception("SAM 失败 %s view_%s: %s", object_id, view_id, e)
                n_fail += 1
                errors.append({"object_id": object_id, "view_id": view_id, "error": str(e)})

    stats = {
        "n_success": n_ok,
        "n_fail": n_fail,
        "n_total_views": n_ok + n_fail,
        "avg_proposals_per_view": round(sum(proposal_counts) / len(proposal_counts), 2) if proposal_counts else 0,
        "errors": errors[:200],
    }
    stats_dir.mkdir(parents=True, exist_ok=True)
    with (stats_dir / "run_sam_proposals_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    log.info("结束: 成功 %d, 失败 %d, 平均 proposals/view %.2f", n_ok, n_fail, stats["avg_proposals_per_view"])


if __name__ == "__main__":
    main()
