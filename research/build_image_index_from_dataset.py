# -*- coding: utf-8 -*-
"""
扫描每个 object 的 imgs/ 目录，建立 40 张图像索引，供伪标签 pipeline 使用。

功能：
- 读取 dataset_index_all.json，得到 object 列表及每个 object 的 imgs_dir
- 对每个 object 扫描 imgs/ 下图像文件，按文件名排序，分配稳定 view_id（0, 1, …, 39）
- 输出每个 object 的 image list 及全局索引，支持小批量验证参数
详见 research/docs/design_pseudo_seg_pipeline.md
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# 默认 7 类（与 scan_dataset 一致）
DEFAULT_CATEGORIES = [
    "Dishwasher",
    "Microwave",
    "Oven",
    "Refrigerator",
    "StorageFurniture",
    "Table",
    "WashingMachine",
]

# 支持的图像扩展名
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def resolve_default_paths() -> Dict[str, Path]:
    """解析默认路径：与 scan_dataset 一致，输出到 pseudo_seg 下。"""
    repo_root = Path(__file__).resolve().parents[1]
    project_root = repo_root.parent / "project"
    processed_root = project_root / "processed_data"
    dataset_index = processed_root / "dataset_index_all.json"
    pseudo_seg_root = processed_root / "pseudo_seg"
    return {
        "processed_root": processed_root,
        "dataset_index": dataset_index,
        "pseudo_seg_root": pseudo_seg_root,
        "indexes_dir": pseudo_seg_root / "indexes",
        "logs_dir": pseudo_seg_root / "logs",
    }


def load_dataset_index(index_path: Path) -> List[Dict[str, Any]]:
    """
    加载 dataset_index_all.json。
    返回列表，每项含 category, model_id, object_dir, imgs_dir 等。
    """
    if not index_path.is_file():
        raise FileNotFoundError(f"数据集索引不存在: {index_path}")
    with index_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("dataset_index_all.json 应为 JSON 数组")
    return data


def list_images_in_dir(imgs_dir: Path, log: logging.Logger) -> List[Path]:
    """
    列出 imgs_dir 下所有图像文件，按文件名排序，保证 view_id 稳定。
    只保留 IMAGE_EXTENSIONS 中的后缀；排序按字符串，使 00.png < 01.png < … < 39.png。
    """
    if not imgs_dir.is_dir():
        return []
    files: List[Path] = []
    for p in imgs_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(p)
    files.sort(key=lambda x: x.name)
    return files


def build_object_image_list(
    object_dir: str,
    imgs_dir: str,
    category: str,
    model_id: str,
    limit_views: Optional[int],
    log: logging.Logger,
) -> Dict[str, Any]:
    """
    为单个 object 构建 image list。
    返回含 object_id, category, model_id, object_dir, imgs_dir, n_views, views 的字典。
    views 中每项含 view_id, image_filename, image_path_rel, image_path_abs。
    若 limit_views 非空，只保留前 limit_views 个 view。
    """
    obj_dir = Path(object_dir).expanduser().resolve()
    imgs_path = Path(imgs_dir).expanduser().resolve()
    if not imgs_path.is_dir():
        # 尝试 fallback：object_dir/imgs
        imgs_path = obj_dir / "imgs"
    image_files = list_images_in_dir(imgs_path, log)
    if not image_files:
        log.warning("无图像: %s", imgs_path)
        return {
            "object_id": f"{category}_{model_id}",
            "category": category,
            "model_id": model_id,
            "object_dir": str(obj_dir),
            "imgs_dir": str(imgs_path),
            "n_views": 0,
            "views": [],
        }
    views: List[Dict[str, Any]] = []
    for view_id, img_path in enumerate(image_files):
        if limit_views is not None and view_id >= limit_views:
            break
        try:
            rel_str = str(img_path.relative_to(obj_dir)).replace("\\", "/")
        except ValueError:
            rel_str = f"imgs/{img_path.name}"
        views.append({
            "view_id": view_id,
            "image_filename": img_path.name,
            "image_path_rel": rel_str,
            "image_path_abs": str(img_path.resolve()),
        })
    return {
        "object_id": f"{category}_{model_id}",
        "category": category,
        "model_id": model_id,
        "object_dir": str(obj_dir),
        "imgs_dir": str(imgs_path),
        "n_views": len(views),
        "views": views,
    }


def setup_logging(logs_dir: Path) -> logging.Logger:
    """配置日志：写文件并输出到控制台。"""
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "build_image_index.log"
    logger = logging.getLogger("build_image_index")
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
    parser = argparse.ArgumentParser(
        description="扫描每个 object 的 imgs/，建立 40 张图索引，供伪标签 pipeline 使用",
    )
    parser.add_argument("--index", type=str, default=str(paths["dataset_index"]), help="dataset_index_all.json 路径")
    parser.add_argument("--output", type=str, default=str(paths["indexes_dir"] / "image_index.json"), help="输出 image_index.json 路径")
    parser.add_argument("--categories", type=str, nargs="*", default=None, help="只处理这些类别，默认全部 7 类")
    parser.add_argument("--limit_per_category", type=int, default=None, help="每类最多处理几个 object，小批量时设为 1")
    parser.add_argument("--limit_views_per_object", type=int, default=None, help="每个 object 最多保留几张图，小批量时设为 3～5")
    parser.add_argument("--debug_single_object", type=str, default=None, help="只处理指定 object_id，例如 Dishwasher_11622")
    parser.add_argument("--dry_run", action="store_true", help="只打印将要处理的 object 与 view 数，不写文件")
    args = parser.parse_args()

    index_path = Path(args.index).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    logs_dir = paths["logs_dir"]
    log = setup_logging(logs_dir)

    log.info("读取索引: %s", index_path)
    records = load_dataset_index(index_path)
    categories = args.categories or DEFAULT_CATEGORIES
    records = [r for r in records if r.get("category") in categories]

    if args.debug_single_object:
        records = [r for r in records if f"{r.get('category')}_{r.get('model_id')}" == args.debug_single_object]
        if not records:
            log.error("未找到 object_id=%s", args.debug_single_object)
            sys.exit(1)
    elif args.limit_per_category is not None:
        from collections import defaultdict
        per_cat = defaultdict(int)
        filtered = []
        for r in records:
            cat = r.get("category", "")
            if per_cat[cat] >= args.limit_per_category:
                continue
            filtered.append(r)
            per_cat[cat] += 1
        records = filtered

    objects_index: List[Dict[str, Any]] = []
    total_views = 0
    for i, rec in enumerate(records):
        cat = rec.get("category", "unknown")
        mid = rec.get("model_id", "unknown")
        obj_id = f"{cat}_{mid}"
        log.info("[%d/%d] %s", i + 1, len(records), obj_id)
        obj_list = build_object_image_list(
            rec.get("object_dir", ""),
            rec.get("imgs_dir", ""),
            cat,
            mid,
            args.limit_views_per_object,
            log,
        )
        objects_index.append(obj_list)
        total_views += obj_list.get("n_views", 0)

    if args.dry_run:
        log.info("dry_run: 共 %d 个 object，%d 张图，不写文件", len(objects_index), total_views)
        for o in objects_index:
            log.info("  %s n_views=%d", o["object_id"], o["n_views"])
        return

    out = {
        "meta": {
            "description": "每个 object 的 40 张图像索引，view_id 与文件名顺序一致，供伪标签 pipeline 使用",
            "n_objects": len(objects_index),
            "n_total_views": total_views,
            "limit_views_per_object": args.limit_views_per_object,
        },
        "objects": objects_index,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log.info("已写入 %s：%d objects，%d views", output_path, len(objects_index), total_views)


if __name__ == "__main__":
    main()
