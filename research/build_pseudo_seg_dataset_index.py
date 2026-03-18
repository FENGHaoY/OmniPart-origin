# -*- coding: utf-8 -*-
"""
汇总 pseudo_masks 与 image_index，生成供 part segmentation 训练使用的 dataset index。

功能：
- 读取 pseudo_seg/indexes/image_index.json 与 pseudo_masks/<category>/<model_id>/view_xxx_pseudo_partseg.png
- 对每个 object 的每个 view，若存在该 view 的 pseudo_partseg 与 meta，则加入索引
- 输出 pseudo_seg/indexes/pseudo_seg_dataset_index.json：样本列表，每项含 object_id, category, model_id,
  view_id, image_path, pseudo_partseg_path, pseudo_meta_path, n_parts 等，供训练脚本读取
- 支持与前述 pipeline 一致的小批量过滤参数
详见 research/docs/design_pseudo_seg_pipeline.md
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    repo_root = Path(__file__).resolve().parents[1]
    project_root = repo_root.parent / "project"
    processed_root = project_root / "processed_data"
    pseudo_seg_root = processed_root / "pseudo_seg"
    return {
        "pseudo_seg_root": pseudo_seg_root,
        "image_index": pseudo_seg_root / "indexes" / "image_index.json",
        "pseudo_masks_dir": pseudo_seg_root / "pseudo_masks",
        "indexes_dir": pseudo_seg_root / "indexes",
        "logs_dir": pseudo_seg_root / "logs",
        "stats_dir": pseudo_seg_root / "stats",
    }


def load_image_index(index_path: Path) -> Dict[str, Any]:
    if not index_path.is_file():
        raise FileNotFoundError(f"image_index 不存在: {index_path}")
    with index_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def setup_logging(logs_dir: Path) -> logging.Logger:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "build_pseudo_seg_dataset_index.log"
    log = logging.getLogger("build_pseudo_seg_dataset_index")
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
    parser = argparse.ArgumentParser(description="汇总 pseudo_masks 与 image_index，生成训练用 pseudo_seg_dataset_index.json")
    parser.add_argument("--image_index", type=str, default=str(paths["image_index"]))
    parser.add_argument("--pseudo_masks_dir", type=str, default=str(paths["pseudo_masks_dir"]))
    parser.add_argument("--output", type=str, default=str(paths["indexes_dir"] / "pseudo_seg_dataset_index.json"))
    parser.add_argument("--categories", type=str, nargs="*", default=None)
    parser.add_argument("--limit_per_category", type=int, default=None)
    parser.add_argument("--limit_views_per_object", type=int, default=None)
    parser.add_argument("--debug_single_object", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    image_index_path = Path(args.image_index).expanduser().resolve()
    pseudo_masks_dir = Path(args.pseudo_masks_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
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

    samples: List[Dict[str, Any]] = []
    for obj in objects:
        object_id = obj.get("object_id", "?")
        category = obj.get("category", "unknown")
        model_id = obj.get("model_id", "unknown")
        views = obj.get("views", [])
        out_dir = pseudo_masks_dir / category / model_id
        if not out_dir.is_dir():
            continue
        for v in views:
            view_id = v.get("view_id", 0)
            partseg_path = out_dir / f"view_{view_id:03d}_pseudo_partseg.png"
            meta_path = out_dir / f"view_{view_id:03d}_pseudo_meta.json"
            if not partseg_path.is_file():
                continue
            image_path = v.get("image_path_abs", "")
            n_parts = None
            part_order = None
            if meta_path.is_file():
                with meta_path.open("r", encoding="utf-8") as f:
                    meta = json.load(f)
                n_parts = meta.get("n_parts")
                part_order = meta.get("part_order")
            samples.append({
                "object_id": object_id,
                "category": category,
                "model_id": model_id,
                "view_id": view_id,
                "image_path": image_path,
                "pseudo_partseg_path": str(partseg_path.resolve()),
                "pseudo_meta_path": str(meta_path.resolve()) if meta_path.is_file() else None,
                "n_parts": n_parts,
                "part_order": part_order,
            })

    out = {
        "meta": {
            "description": "Part segmentation 伪标签训练用索引，由 build_pseudo_seg_dataset_index.py 生成",
            "n_samples": len(samples),
            "source_image_index": str(image_index_path),
            "source_pseudo_masks_dir": str(pseudo_masks_dir),
        },
        "samples": samples,
    }

    if args.dry_run:
        log.info("dry_run: 将写入 %d 条样本到 %s", len(samples), output_path)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log.info("已写入 %s：%d 条样本", output_path, len(samples))

    stats_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        "n_samples": len(samples),
        "output_path": str(output_path),
        "n_objects_with_any_view": len({s["object_id"] for s in samples}),
    }
    with (stats_dir / "build_pseudo_seg_dataset_index_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
