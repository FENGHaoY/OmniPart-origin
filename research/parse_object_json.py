# -*- coding: utf-8 -*-
"""
解析 object.json 中的 diffuse_tree，生成标准化的 object_meta.json。

目标：为「逻辑部件级 mask 渲染」与「joint-aware bbox token 监督」提供统一、
稳定、可解释的数据格式。详见 research/docs/design_parse_object_json.md 与 README_object_meta.md。
"""

import argparse
import hashlib
import json
import logging
import random
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# -----------------------------------------------------------------------------
# 路径与配置
# -----------------------------------------------------------------------------


def resolve_default_paths() -> Dict[str, Path]:
    """
    解析默认输入/输出路径（与 scan_dataset.py 一致）。
    - 输入：../project/processed_data/dataset_index_all.json
    - 输出根：../project/processed_data/
    """
    repo_root = Path(__file__).resolve().parents[1]
    project_root = repo_root.parent / "project"
    processed_root = project_root / "processed_data"
    index_path = processed_root / "dataset_index_all.json"
    return {
        "dataset_index": index_path,
        "processed_root": processed_root,
        "objects_meta_root": processed_root / "objects_meta",
        "indexes_dir": processed_root / "indexes",
        "logs_dir": processed_root / "logs",
        "stats_dir": processed_root / "stats",
        "debug_dir": processed_root / "debug_samples",
    }


# -----------------------------------------------------------------------------
# 加载 dataset index
# -----------------------------------------------------------------------------


def load_dataset_index(index_path: Path) -> List[Dict[str, Any]]:
    """
    加载 dataset_index_all.json。
    返回：列表，每项为一条样本（含 category, model_id, object_dir, object_json 等）。
    """
    if not index_path.is_file():
        raise FileNotFoundError(f"数据集索引不存在: {index_path}")
    with index_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("dataset_index_all.json 应为 JSON 数组")
    return data


# -----------------------------------------------------------------------------
# 从 object.json 获取 diffuse_tree 根节点
# -----------------------------------------------------------------------------


def get_diffuse_tree(obj: Dict[str, Any]) -> Any:
    """
    从已解析的 object.json 中取出 diffuse_tree。
    - 若存在 "diffuse_tree" 键：其值可能是 list（part 节点数组）或 dict（单根节点）。
    - 否则将整份 JSON 视为根节点（dict）。
    返回 (root_or_list, is_list)。is_list 为 True 表示 Singapo 格式：diffuse_tree 为 part 数组，每节点含 id/parent/children。
    """
    if "diffuse_tree" in obj:
        root = obj["diffuse_tree"]
    else:
        root = obj
    if isinstance(root, list):
        return (root, True)
    if isinstance(root, dict):
        return (root, False)
    raise ValueError("diffuse_tree 应为 list（part 数组）或 dict（根节点），当前为 %s" % type(root).__name__)


# -----------------------------------------------------------------------------
# diffuse_tree 为 list 时：直接按 id/parent/children 解析（Singapo 格式）
# -----------------------------------------------------------------------------


def parse_diffuse_tree_list(
    node_list: List[Dict[str, Any]],
    log: logging.Logger,
) -> List[Dict[str, Any]]:
    """
    diffuse_tree 为 part 数组时：每节点已有 id、parent、children。
    按 id 排序后依次标准化为 parts[]，part_id 使用节点自带的 id。
    """
    if not node_list:
        return []
    # 按 id 排序，保证顺序稳定
    sorted_nodes = sorted(node_list, key=lambda n: n.get("id", 0))
    parts = []
    for node in sorted_nodes:
        part_id = node.get("id", 0)
        parent_id = node.get("parent", -1)
        children = node.get("children") or []
        if not isinstance(children, list):
            children = [children] if children else []
        part = normalize_part_node(node, part_id, parent_id, children, log)
        parts.append(part)
    return parts


# -----------------------------------------------------------------------------
# 树遍历：BFS 得到 (node, part_id, parent_part_id) 列表（dict 根节点格式）
# -----------------------------------------------------------------------------


def flatten_tree_bfs(root: Dict[str, Any]) -> List[Tuple[Dict[str, Any], int, int]]:
    """
    BFS 遍历部件树，为每个节点分配 part_id（从 0 开始），并记录其父节点 part_id。
    返回：[(node, part_id, parent_part_id), ...]，其中根节点的 parent_part_id = -1。
    """
    result: List[Tuple[Dict[str, Any], int, int]] = []
    # 队列元素：(node, parent_part_id)
    queue: deque = deque([(root, -1)])
    part_id = 0
    while queue:
        node, parent_part_id = queue.popleft()
        result.append((node, part_id, parent_part_id))
        children = node.get("children") or node.get("child") or []
        if not isinstance(children, list):
            children = [children] if children else []
        for ch in children:
            if isinstance(ch, dict):
                queue.append((ch, part_id))
        part_id += 1
    return result


# -----------------------------------------------------------------------------
# 单个 part 节点标准化：joint / bbox / mesh_files / ply_files 等
# -----------------------------------------------------------------------------


def _parse_joint(node: Dict[str, Any], part_id: int, log: logging.Logger) -> Dict[str, Any]:
    """
    从节点解析 joint 字段，缺省时使用 fixed 与零向量，并打日志。
    """
    raw = node.get("joint") or node.get("joint_info")
    if not raw or not isinstance(raw, dict):
        return {
            "type": "fixed",
            "range": [0.0, 0.0],
            "axis_origin": [0.0, 0.0, 0.0],
            "axis_direction": [0.0, 0.0, 0.0],
        }
    jtype = (raw.get("type") or raw.get("joint_type") or "fixed")
    if isinstance(jtype, str):
        jtype = jtype.strip().lower()
    else:
        jtype = "fixed"
    # 支持 continuous（连续旋转，与 revolute 同类）
    if jtype not in ("fixed", "revolute", "prismatic", "continuous", "none", ""):
        log.debug("part_id=%s 未知 joint type %s，视为 fixed", part_id, jtype)
        jtype = "fixed"
    if jtype in ("none", ""):
        jtype = "fixed"
    # 支持嵌套 joint.axis: { "origin": [...], "direction": [...] }（Singapo 格式）
    axis = raw.get("axis")
    if isinstance(axis, dict):
        origin = axis.get("origin") or axis.get("pivot") or [0.0, 0.0, 0.0]
        direction = axis.get("direction") or axis.get("axis") or [0.0, 0.0, 0.0]
    else:
        origin = raw.get("axis_origin") or raw.get("origin") or raw.get("pivot") or [0.0, 0.0, 0.0]
        direction = raw.get("axis_direction") or raw.get("axis") or raw.get("direction") or [0.0, 0.0, 0.0]
    rng = raw.get("range") or raw.get("joint_range") or [0.0, 0.0]
    if not isinstance(origin, (list, tuple)) or len(origin) != 3:
        log.debug("part_id=%s axis_origin 格式异常，用 [0,0,0]", part_id)
        origin = [0.0, 0.0, 0.0]
    if not isinstance(direction, (list, tuple)) or len(direction) != 3:
        log.debug("part_id=%s axis_direction 格式异常，用 [0,0,0]", part_id)
        direction = [0.0, 0.0, 0.0]
    if not isinstance(rng, (list, tuple)) or len(rng) < 2:
        rng = [0.0, 0.0]
    return {
        "type": jtype,
        "range": [float(rng[0]), float(rng[1])],
        "axis_origin": [float(origin[0]), float(origin[1]), float(origin[2])],
        "axis_direction": [float(direction[0]), float(direction[1]), float(direction[2])],
    }


def _parse_bbox(node: Dict[str, Any], part_id: int, log: logging.Logger) -> Dict[str, List[float]]:
    """
    从节点解析 bbox，支持 bbox、aabb（Singapo 用 aabb.center/size）、min/max、bbox_min/max。
    缺失时返回 center=[0,0,0], size=[0,0,0] 并打日志。
    """
    # Singapo 格式使用 aabb: { center, size }
    bbox = node.get("aabb") or node.get("bbox")
    if isinstance(bbox, dict):
        if "min" in bbox and "max" in bbox:
            mn = bbox["min"]
            mx = bbox["max"]
            if len(mn) == 3 and len(mx) == 3:
                center = [(mn[i] + mx[i]) / 2.0 for i in range(3)]
                size = [max(0.0, mx[i] - mn[i]) for i in range(3)]
                return {"center": center, "size": size}
        if "center" in bbox and "size" in bbox:
            return {"center": list(bbox["center"]), "size": list(bbox["size"])}
    mn = node.get("bbox_min") or node.get("min")
    mx = node.get("bbox_max") or node.get("max")
    if mn is not None and mx is not None and len(mn) == 3 and len(mx) == 3:
        center = [(mn[i] + mx[i]) / 2.0 for i in range(3)]
        size = [max(0.0, mx[i] - mn[i]) for i in range(3)]
        return {"center": center, "size": size}
    log.debug("part_id=%s 无有效 bbox，使用默认 [0,0,0]", part_id)
    return {"center": [0.0, 0.0, 0.0], "size": [0.0, 0.0, 0.0]}


def _parse_mesh_ply_list(node: Dict[str, Any], key_objs: str, key_plys: str) -> Tuple[List[str], List[str]]:
    """
    从节点取 objs / plys 列表；可能键名为 objs/obj、plys/ply。
    返回 (mesh_files, ply_files)，均为相对路径字符串列表。
    """
    objs = node.get(key_objs) or node.get("obj") or node.get("mesh_files") or []
    plys = node.get(key_plys) or node.get("ply") or node.get("ply_files") or []
    if not isinstance(objs, list):
        objs = [objs] if objs else []
    if not isinstance(plys, list):
        plys = [plys] if plys else []
    mesh_files = [str(x).strip() for x in objs if x]
    ply_files = [str(x).strip() for x in plys if x]
    return mesh_files, ply_files


def normalize_part_node(
    node: Dict[str, Any],
    part_id: int,
    parent_id: int,
    children: List[int],
    log: logging.Logger,
) -> Dict[str, Any]:
    """
    将 diffuse_tree 的一个 node 标准化为 object_meta 中 parts[] 的一项。
    包含：part_id, parent_id, children, name, joint, bbox, mesh_files, ply_files, is_articulated, is_leaf。
    """
    name = node.get("name") or node.get("id") or f"part_{part_id}"
    if not isinstance(name, str):
        name = str(name)
    joint = _parse_joint(node, part_id, log)
    bbox = _parse_bbox(node, part_id, log)
    mesh_files, ply_files = _parse_mesh_ply_list(node, "objs", "plys")
    jtype = (joint.get("type") or "fixed").lower()
    is_articulated = jtype not in ("fixed", "none", "")
    is_leaf = len(children) == 0
    return {
        "part_id": part_id,
        "parent_id": parent_id,
        "children": children,
        "name": name,
        "joint": joint,
        "bbox": bbox,
        "mesh_files": mesh_files,
        "ply_files": ply_files,
        "is_articulated": is_articulated,
        "is_leaf": is_leaf,
    }


# -----------------------------------------------------------------------------
# 构建 part_order（第一版：按 part_id 升序）
# -----------------------------------------------------------------------------


def build_part_order(parts: List[Dict[str, Any]]) -> List[int]:
    """
    第一版规则：part_order = sorted(all part_id)。
    这样与 BFS 分配的 part_id 一致，实现简单、便于复现。若后续要改为 BFS/DFS 序，
    只需在此处改为按遍历顺序收集 part_id 即可。
    """
    return sorted(p["part_id"] for p in parts)


# -----------------------------------------------------------------------------
# 构建 meta：obj_cat, depth, n_arti_parts, n_revolute, n_prismatic, n_diff_parts, tree_hash
# -----------------------------------------------------------------------------


def _tree_depth(parts: List[Dict[str, Any]]) -> int:
    """从根出发的最大深度（根深度为 1）。"""
    if not parts:
        return 0
    id_to_part = {p["part_id"]: p for p in parts}
    depth = 1
    stack = [(0, 1)]  # (part_id, depth)
    while stack:
        pid, d = stack.pop()
        depth = max(depth, d)
        for c in id_to_part.get(pid, {}).get("children", []):
            stack.append((c, d + 1))
    return depth


def _tree_hash(parts: List[Dict[str, Any]]) -> str:
    """用于可复现的结构哈希：part_id -> sorted(children)。"""
    id_to_children = {p["part_id"]: sorted(p["children"]) for p in parts}
    s = json.dumps(id_to_children, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def build_meta(
    parts: List[Dict[str, Any]],
    raw_root: Dict[str, Any],
    obj_cat: str,
) -> Dict[str, Any]:
    """
    汇总物体级统计与摘要，便于追溯与统计。
    """
    n_diff_parts = len(parts)
    n_revolute = sum(1 for p in parts if (p.get("joint") or {}).get("type") == "revolute")
    n_prismatic = sum(1 for p in parts if (p.get("joint") or {}).get("type") == "prismatic")
    n_arti_parts = sum(1 for p in parts if p.get("is_articulated"))
    depth = _tree_depth(parts)
    tree_hash = _tree_hash(parts)
    return {
        "obj_cat": obj_cat,
        "depth": depth,
        "n_arti_parts": n_arti_parts,
        "n_revolute": n_revolute,
        "n_prismatic": n_prismatic,
        "n_diff_parts": n_diff_parts,
        "tree_hash": tree_hash,
    }


# -----------------------------------------------------------------------------
# 解析单个 object.json，返回标准化 object_meta 字典
# -----------------------------------------------------------------------------


def parse_single_object_json(
    object_json_path: Path,
    object_dir: Path,
    category: str,
    model_id: str,
    log: logging.Logger,
) -> Dict[str, Any]:
    """
    读取并解析一份 object.json，返回可写入 object_meta.json 的字典。
    若解析失败则抛出异常，由调用方捕获并记录。
    """
    if not object_json_path.is_file():
        raise FileNotFoundError(f"object.json 不存在: {object_json_path}")
    with object_json_path.open("r", encoding="utf-8") as f:
        raw_obj = json.load(f)
    tree_val, is_list = get_diffuse_tree(raw_obj)
    if is_list:
        # Singapo 格式：diffuse_tree 为 part 数组，每节点含 id/parent/children
        parts = parse_diffuse_tree_list(tree_val, log)
        raw_root_for_meta = raw_obj.get("meta") or {}
    else:
        # 单根 dict：BFS 展开
        flat = flatten_tree_bfs(tree_val)
        if not flat:
            raise ValueError("diffuse_tree 为空")
        part_children: Dict[int, List[int]] = {t[1]: [] for t in flat}
        for node, part_id, parent_part_id in flat:
            if parent_part_id >= 0:
                part_children[parent_part_id].append(part_id)
        parts = []
        for node, part_id, parent_part_id in flat:
            children = part_children.get(part_id, [])
            part = normalize_part_node(node, part_id, parent_part_id, children, log)
            parts.append(part)
        raw_root_for_meta = tree_val
    if not parts:
        raise ValueError("diffuse_tree 解析后无 part")
    part_order = build_part_order(parts)
    obj_cat = raw_obj.get("meta", {}).get("obj_cat") or raw_obj.get("obj_cat") or raw_obj.get("category") or category
    if not isinstance(obj_cat, str):
        obj_cat = category
    meta = build_meta(parts, raw_root_for_meta, obj_cat)
    object_id = f"{category}_{model_id}"
    return {
        "object_id": object_id,
        "category": category,
        "model_id": model_id,
        "source_object_dir": str(object_dir.resolve()),
        "source_object_json": str(object_json_path.resolve()),
        "meta": meta,
        "part_order": part_order,
        "parts": parts,
    }


# -----------------------------------------------------------------------------
# 保存 object_meta.json 与总索引
# -----------------------------------------------------------------------------


def save_object_meta(meta_dict: Dict[str, Any], out_path: Path, log: logging.Logger) -> None:
    """将标准化 object_meta 写入 JSON 文件。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2, ensure_ascii=False)
    log.info("已写入 object_meta: %s", out_path)


def collect_statistics(
    success_list: List[Dict[str, Any]],
    fail_list: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    汇总成功/失败数、每类样本数、每类平均 part 数、各 joint type 计数。
    """
    per_cat: Dict[str, Dict[str, Any]] = {}
    joint_type_counts: Dict[str, int] = {}
    for meta in success_list:
        cat = meta.get("category", "unknown")
        if cat not in per_cat:
            per_cat[cat] = {"count": 0, "n_parts_sum": 0}
        per_cat[cat]["count"] += 1
        per_cat[cat]["n_parts_sum"] += len(meta.get("parts") or [])
        for p in meta.get("parts") or []:
            jt = (p.get("joint") or {}).get("type", "fixed")
            joint_type_counts[jt] = joint_type_counts.get(jt, 0) + 1
    summary: Dict[str, Any] = {
        "n_success": len(success_list),
        "n_fail": len(fail_list),
        "per_category": {
            cat: {
                "count": info["count"],
                "avg_parts": round(info["n_parts_sum"] / info["count"], 2) if info["count"] else 0,
            }
            for cat, info in per_cat.items()
        },
        "joint_type_counts": joint_type_counts,
        "errors": [{"category": e.get("category"), "model_id": e.get("model_id"), "error": e.get("error")} for e in fail_list],
    }
    return summary


# -----------------------------------------------------------------------------
# 主流程：读取 index，逐个解析，写 object_meta、索引、日志、统计、debug
# -----------------------------------------------------------------------------


def setup_logging(log_dir: Path) -> logging.Logger:
    """配置同时输出到文件与控制台的 logger。"""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "parse_object_json.log"
    logger = logging.getLogger("parse_object_json")
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
    parser = argparse.ArgumentParser(description="解析 object.json 的 diffuse_tree，生成 object_meta.json")
    parser.add_argument("--index", type=str, default=str(paths["dataset_index"]), help="dataset_index_all.json 路径")
    parser.add_argument("--processed_root", type=str, default=str(paths["processed_root"]), help="processed_data 根目录")
    args = parser.parse_args()
    index_path = Path(args.index).expanduser().resolve()
    processed_root = Path(args.processed_root).expanduser().resolve()
    objects_meta_root = processed_root / "objects_meta"
    indexes_dir = processed_root / "indexes"
    logs_dir = processed_root / "logs"
    stats_dir = processed_root / "stats"
    debug_dir = processed_root / "debug_samples"

    log = setup_logging(logs_dir)
    log.info("开始解析 object.json，索引: %s", index_path)

    records = load_dataset_index(index_path)
    log.info("共加载 %d 条样本", len(records))

    object_meta_index: List[Dict[str, str]] = []
    success_list: List[Dict[str, Any]] = []
    fail_list: List[Dict[str, Any]] = []

    for i, rec in enumerate(records):
        category = rec.get("category", "unknown")
        model_id = rec.get("model_id", "unknown")
        object_dir = Path(rec.get("object_dir", ""))
        object_json_str = rec.get("object_json", "")
        object_json_path = Path(object_json_str) if object_json_str else object_dir / "object.json"
        log.info("[%d/%d] 解析 %s / %s", i + 1, len(records), category, model_id)
        try:
            meta_dict = parse_single_object_json(object_json_path, object_dir, category, model_id, log)
            out_path = objects_meta_root / category / model_id / "object_meta.json"
            save_object_meta(meta_dict, out_path, log)
            object_meta_index.append({
                "category": category,
                "model_id": model_id,
                "object_id": meta_dict["object_id"],
                "object_meta_path": str(out_path.resolve()),
            })
            success_list.append(meta_dict)
        except Exception as e:
            log.exception("解析失败 %s / %s: %s", category, model_id, e)
            fail_list.append({"category": category, "model_id": model_id, "error": str(e)})

    # 总索引
    indexes_dir.mkdir(parents=True, exist_ok=True)
    index_out = indexes_dir / "object_meta_index.json"
    with index_out.open("w", encoding="utf-8") as f:
        json.dump(object_meta_index, f, indent=2, ensure_ascii=False)
    log.info("已写入总索引: %s，共 %d 条", index_out, len(object_meta_index))

    # 统计
    stats = collect_statistics(success_list, fail_list)
    stats_dir.mkdir(parents=True, exist_ok=True)
    stats_path = stats_dir / "object_meta_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    log.info("已写入统计: %s", stats_path)
    log.info("成功 %d，失败 %d", stats["n_success"], stats["n_fail"])

    # 随机 3～5 个样本写入 debug_samples，包含完整 object_meta 与简要说明
    if success_list:
        n_debug = min(5, max(3, len(success_list)))
        debug_samples = random.sample(success_list, n_debug)
        debug_dir.mkdir(parents=True, exist_ok=True)
        for idx, meta in enumerate(debug_samples):
            name = f"{meta['category']}_{meta['model_id']}_debug_{idx}.json"
            with (debug_dir / name).open("w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
        readme = debug_dir / "README.txt"
        with readme.open("w", encoding="utf-8") as f:
            f.write("本目录为 parse_object_json 随机挑选的 %d 个样本的完整 object_meta，便于人工检查解析结果。\n" % n_debug)
        log.info("已写入 %d 个 debug 样本到 %s", n_debug, debug_dir)

    log.info("parse_object_json 完成。")


if __name__ == "__main__":
    main()
