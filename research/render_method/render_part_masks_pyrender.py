# -*- coding: utf-8 -*-
"""
使用 trimesh + pyrender 离屏渲染 object_meta 的多视角 RGB、part segmentation mask、depth。

渲染单位是逻辑 part（一个 part 可对应多个 mesh 文件，合并后渲染为同一 mask id）。
视角采用前半球固定模板；相机距离根据 overall bbox 自动计算。
小批量验证：默认每类 1 个 object、每 object 3 个视角。
详见 research/docs/design_render_part_masks.md 与 README_render_part_masks.md。
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# 离屏渲染：无显示器时使用 EGL（aarch64/服务器常见）
if "PYOPENGL_PLATFORM" not in os.environ:
    os.environ["PYOPENGL_PLATFORM"] = "egl"

try:
    import trimesh
except ImportError:
    raise ImportError("请安装 trimesh: pip install trimesh")

try:
    import pyrender
except ImportError:
    raise ImportError("请安装 pyrender: pip install pyrender（无头服务器可能需 EGL）")

# -----------------------------------------------------------------------------
# 路径与配置
# -----------------------------------------------------------------------------

# 默认 7 类（与 scan_dataset / parse_object_json 一致）
DEFAULT_CATEGORIES = [
    "Dishwasher",
    "Microwave",
    "Oven",
    "Refrigerator",
    "StorageFurniture",
    "Table",
    "WashingMachine",
]

# 前半球视角：azimuth 与 elevation（度），共 7*3=21 个视角
# 只渲染正面前半球，因物体多有 canonical front、articulation 在正面，背面对 part seg 意义小
DEFAULT_AZIMUTH_DEG = [-75, -50, -25, 0, 25, 50, 75]
DEFAULT_ELEVATION_DEG = [15, 35, 55]

# 相机距离：基于 bbox 的 scale 与 margin，保证物体完整入画
BBOX_DISTANCE_FACTOR = 1.8
BBOX_DISTANCE_MARGIN = 0.5

# 渲染分辨率
DEFAULT_IMAGE_WIDTH = 256
DEFAULT_IMAGE_HEIGHT = 256


def resolve_default_paths() -> Dict[str, Path]:
    """解析默认输入/输出路径（与 parse_object_json 一致）。"""
    repo_root = Path(__file__).resolve().parents[1]
    project_root = repo_root.parent / "project"
    processed_root = project_root / "processed_data"
    return {
        "processed_root": processed_root,
        "object_meta_index": processed_root / "indexes" / "object_meta_index.json",
        "renders_root": processed_root / "renders",
        "logs_dir": processed_root / "logs",
        "stats_dir": processed_root / "stats",
        "debug_renders_dir": processed_root / "debug_renders",
    }


# -----------------------------------------------------------------------------
# 加载 object_meta 与 part meshes
# -----------------------------------------------------------------------------


def load_object_meta(object_meta_path: Path) -> Dict[str, Any]:
    """
    加载单份 object_meta.json。
    返回标准化 object_meta 字典（含 object_id, category, model_id, source_object_dir, parts 等）。
    """
    if not object_meta_path.is_file():
        raise FileNotFoundError(f"object_meta 不存在: {object_meta_path}")
    with object_meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_part_meshes(
    object_meta: Dict[str, Any],
    log: logging.Logger,
    fallback_source_dir: Optional[Path] = None,
) -> List[Tuple[int, trimesh.Trimesh]]:
    """
    按 object_meta 的 parts 加载每个逻辑 part 的 mesh（优先 mesh_files，否则 ply_files）。
    mesh_files 为相对路径，相对 base_dir；base_dir 优先用 object_meta 的 source_object_dir（做 resolve），
    若不存在则用 fallback_source_dir（通常为 project/data/<category>/<model_id>），便于不同环境复现。
    一个 part 的多个 mesh 合并为一个 trimesh。
    返回 [(part_id, trimesh), ...]，无有效 mesh 的 part 会跳过并打日志。
    """
    raw_dir = object_meta.get("source_object_dir", "")
    source_dir = Path(raw_dir).expanduser().resolve() if raw_dir else Path()
    if not source_dir.is_dir() and fallback_source_dir is not None and fallback_source_dir.is_dir():
        log.info("source_object_dir 不存在 %s，使用 fallback: %s", source_dir, fallback_source_dir)
        source_dir = fallback_source_dir
    if not source_dir.is_dir():
        log.warning("source_object_dir 不存在且无可用 fallback: %s", raw_dir)
        return []

    log.debug("加载 mesh 的 base 目录: %s", source_dir)
    result: List[Tuple[int, trimesh.Trimesh]] = []
    for part in object_meta.get("parts", []):
        part_id = part.get("part_id", -1)
        mesh_files = part.get("mesh_files") or []
        if not mesh_files:
            mesh_files = part.get("ply_files") or []
        if not mesh_files:
            log.debug("part_id=%s 无 mesh_files/ply_files，跳过", part_id)
            continue

        meshes: List[trimesh.Trimesh] = []
        for rel_path in mesh_files:
            # 相对路径统一按 source_dir 解析，避免相对路径与当前工作目录混淆
            rel_path_str = str(rel_path).lstrip("/")
            path = (source_dir / rel_path_str).resolve()
            if not path.is_file():
                log.debug("mesh 文件不存在: %s (base=%s)", path, source_dir)
                continue
            try:
                m = trimesh.load(path, force="mesh", process=False)
                if m is None:
                    continue
                if isinstance(m, trimesh.Scene):
                    for g in m.geometry.values():
                        if isinstance(g, trimesh.Trimesh):
                            meshes.append(g)
                elif isinstance(m, trimesh.Trimesh):
                    meshes.append(m)
            except Exception as e:
                log.debug("加载 mesh 失败 %s: %s", path, e)

        if not meshes:
            log.warning("part_id=%s 无有效 mesh，跳过 (base_dir=%s)", part_id, source_dir)
            continue
        try:
            merged = trimesh.util.concatenate(meshes)
            result.append((part_id, merged))
        except Exception as e:
            log.warning("part_id=%s 合并 mesh 失败: %s", part_id, e)
    return result


def compute_overall_bbox(
    part_meshes: List[Tuple[int, trimesh.Trimesh]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据所有 part 的 trimesh 顶点计算整体 AABB。
    返回 (center, size)，均为 shape (3,)。
    """
    if not part_meshes:
        return np.zeros(3), np.zeros(3)
    all_verts = np.vstack([m.vertices for _, m in part_meshes])
    min_ = all_verts.min(axis=0)
    max_ = all_verts.max(axis=0)
    center = (min_ + max_) * 0.5
    size = np.maximum(max_ - min_, 1e-6)
    return center.astype(np.float64), size.astype(np.float64)


# -----------------------------------------------------------------------------
# 前半球视角模板
# -----------------------------------------------------------------------------


def generate_camera_views(
    azimuth_deg_list: Sequence[float],
    elevation_deg_list: Sequence[float],
    num_views: Optional[int] = None,
) -> List[Tuple[float, float]]:
    """
    生成前半球视角列表：(azimuth_deg, elevation_deg)。
    azimuth 0 对应“正面”方向（与后续相机球坐标一致），背面不渲染。
    若指定 num_views，只返回前 num_views 个（用于小批量验证）。
    """
    views = []
    for el in elevation_deg_list:
        for az in azimuth_deg_list:
            views.append((float(az), float(el)))
    if num_views is not None and num_views < len(views):
        views = views[:num_views]
    return views


def camera_position_from_spherical(
    azimuth_deg: float,
    elevation_deg: float,
    distance: float,
) -> np.ndarray:
    """
    由方位角、仰角、距离得到相机位置（看向原点）。
    约定：azimuth 在 XY 平面，elevation 为与 XY 平面夹角；相机在球面上。
    """
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    x = distance * math.cos(el) * math.cos(az)
    y = distance * math.cos(el) * math.sin(az)
    z = distance * math.sin(el)
    return np.array([x, y, z], dtype=np.float64)


# -----------------------------------------------------------------------------
# 单视角渲染：RGB + part mask（同机位）
# -----------------------------------------------------------------------------


def _make_pyrender_mesh(
    mesh: trimesh.Trimesh,
    color: Tuple[float, float, float, float],
) -> pyrender.Mesh:
    """将 trimesh 转为 pyrender.Mesh，指定颜色。"""
    mat = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=(color[0], color[1], color[2], color[3]),
        metallicFactor=0.0,
        roughnessFactor=1.0,
    )
    return pyrender.Mesh.from_trimesh(mesh, material=mat)


def render_single_view(
    part_meshes: List[Tuple[int, trimesh.Trimesh]],
    center: np.ndarray,
    azimuth_deg: float,
    elevation_deg: float,
    distance: float,
    width: int,
    height: int,
    part_order: List[int],
    render_depth: bool,
    log: logging.Logger,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict[int, Tuple[float, float, float, float]]]:
    """
    渲染单视角：RGB 图、part mask 图、可选 depth、以及 part_id -> 用于 mask 的 (R,G,B,A)。
    所有 mesh 已平移到以 center 为原点（调用方负责先平移）；相机在球坐标 (azimuth, elevation, distance) 看向原点。
    返回 (rgb, part_mask_int, depth_or_None, part_id_to_color_for_mask)。
    part_mask_int: 0=背景，1..N=part_id+1。
    """
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])
    # 方向光，便于看清形状
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    light_pose = np.eye(4)
    light_pose[:3, 3] = [2, 2, 2]
    scene.add(light, pose=light_pose)

    # 为每个 part 分配 RGB 用的中性色（略区分）和 mask 用唯一色
    part_id_to_rgb_color: Dict[int, Tuple[float, float, float]] = {}
    part_id_to_mask_color: Dict[int, Tuple[float, float, float, float]] = {}
    # 简单中性色：按 part_id 在 [0.4, 0.9] 间线性
    for i, (part_id, _) in enumerate(part_meshes):
        t = i / max(len(part_meshes), 1)
        gray = 0.4 + 0.5 * t
        part_id_to_rgb_color[part_id] = (gray, gray * 0.95, gray * 0.9)
        # mask：R = (part_id+1)/255，G=B=0，便于解码
        v = (part_id + 1) / 255.0
        part_id_to_mask_color[part_id] = (v, 0.0, 0.0, 1.0)

    # 先渲染 RGB：用中性色
    for part_id, mesh in part_meshes:
        r, g, b = part_id_to_rgb_color[part_id]
        pm = _make_pyrender_mesh(mesh, (r, g, b, 1.0))
        scene.add(pm)

    cam = pyrender.PerspectiveCamera(yfov=math.radians(45.0), aspectRatio=width / height, znear=0.01, zfar=100.0)
    cam_pose = np.eye(4)
    cam_pose[:3, 3] = camera_position_from_spherical(azimuth_deg, elevation_deg, distance)
    # 相机朝向原点：OpenGL/pyrender 中相机沿 -Z 看，故相机局部 +Z 应指向“从原点指向相机”
    view_dir = -cam_pose[:3, 3]
    view_dir /= np.linalg.norm(view_dir)
    world_up = np.array([0.0, 1.0, 0.0])  # Y-up 与常见 3D/OpenGL 一致
    x_axis = np.cross(world_up, view_dir)
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.cross([1, 0, 0], view_dir)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(view_dir, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    # 相机 +Z = 从原点指向相机（即 -view_dir），这样 -Z 才指向场景
    cam_pose[:3, 0] = x_axis
    cam_pose[:3, 1] = y_axis
    cam_pose[:3, 2] = -view_dir
    scene.add(cam, pose=cam_pose)

    try:
        r = pyrender.OffscreenRenderer(width, height)
        try:
            color_rgb, depth_py = r.render(scene, flags=pyrender.RenderFlags.RGBA)
        finally:
            r.delete()
    except Exception as e:
        log.warning("pyrender RGB 渲染失败: %s", e)
        raise

    # 再渲染 mask：同一机位、同一几何，但颜色为 mask 色，关闭光照影响
    scene_mask = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0])
    for part_id, mesh in part_meshes:
        c = part_id_to_mask_color[part_id]
        pm = _make_pyrender_mesh(mesh, c)
        scene_mask.add(pm)
    cam_mask = pyrender.PerspectiveCamera(yfov=math.radians(45.0), aspectRatio=width / height, znear=0.01, zfar=100.0)
    scene_mask.add(cam_mask, pose=cam_pose)
    try:
        r2 = pyrender.OffscreenRenderer(width, height)
        try:
            # 使用 FLAT 着色，避免插值带来的颜色偏移；仅保留 RGBA 与 FLAT 标志
            flags = pyrender.RenderFlags.FLAT | pyrender.RenderFlags.RGBA
            color_mask, _ = r2.render(scene_mask, flags=flags)
        finally:
            r2.delete()
    except Exception as e:
        log.warning("pyrender mask 渲染失败: %s", e)
        raise

    # 解码 mask：R 通道即 (part_id+1)/255，四舍五入得 0..N
    part_mask_int = decode_part_mask(color_mask, part_id_to_mask_color, part_order, log)

    depth_out = None
    if render_depth and depth_py is not None:
        depth_out = np.copy(depth_py)

    # RGB 只取 RGB 三通道
    rgb = color_rgb[:, :, :3]
    if rgb.dtype != np.uint8:
        rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    return rgb, part_mask_int, depth_out, part_id_to_mask_color


def decode_part_mask(
    color_rgba: np.ndarray,
    part_id_to_mask_color: Dict[int, Tuple[float, float, float, float]],
    part_order: List[int],
    log: logging.Logger,
) -> np.ndarray:
    """
    从 mask 渲染的 color pass 解码为整型 mask：0=背景，1..N=part_id+1。
    本实现中 mask 颜色为 ( (part_id+1)/255, 0, 0, 1 )，用 R 通道四舍五入即可。
    """
    h, w = color_rgba.shape[0], color_rgba.shape[1]
    if color_rgba.max() <= 1.0:
        r_ch = (color_rgba[:, :, 0] * 255.0).astype(np.float64)
    else:
        r_ch = color_rgba[:, :, 0].astype(np.float64)
    # 四舍五入得到 0, 1, 2, ...，理论上应等于 part_id+1
    raw = np.round(r_ch).astype(np.int32)
    # debug：打印前若干 unique 值，便于排查编码/解码是否对齐
    uniq = np.unique(raw)
    log.debug("mask R 通道 unique 值（前 20 个）: %s", uniq[:20])
    out = np.zeros((h, w), dtype=np.int32)
    # 直接使用 raw 作为 mask（0=背景，1..N=part_id+1），并裁剪到合理范围
    max_val = max(part_id + 1 for part_id in part_order) if part_order else 0
    if max_val > 0:
        out = np.clip(raw, 0, max_val)
    else:
        out = raw
    return out


def compute_visible_parts_and_2d_bboxes(
    part_mask: np.ndarray,
) -> Tuple[List[int], Dict[str, List[int]]]:
    """
    从 part_mask（0=背景，1..N=part_id+1）统计可见 part 与每个 part 的 2D bbox [xmin, ymin, xmax, ymax]。
    返回 (visible_parts, part_bboxes_2d)，part_bboxes_2d 的 key 为 "0","1",...（part_id 字符串）。
    """
    visible: List[int] = []
    bboxes: Dict[str, List[int]] = {}
    unique = np.unique(part_mask)
    for val in unique:
        if val == 0:
            continue
        part_id = int(val) - 1
        visible.append(part_id)
        ys, xs = np.where(part_mask == val)
        if ys.size == 0 or xs.size == 0:
            continue
        xmin, xmax = int(xs.min()), int(xs.max())
        ymin, ymax = int(ys.min()), int(ys.max())
        bboxes[str(part_id)] = [xmin, ymin, xmax, ymax]
    return visible, bboxes


# -----------------------------------------------------------------------------
# 保存单视角结果与 meta
# -----------------------------------------------------------------------------


def save_view_outputs(
    out_dir: Path,
    view_id: int,
    rgb: np.ndarray,
    part_mask: np.ndarray,
    depth: Optional[np.ndarray],
    view_meta: Dict[str, Any],
    log: logging.Logger,
) -> None:
    """将单视角的 rgb、partseg、depth、meta 写入 out_dir。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"view_{view_id:03d}"
    # RGB
    try:
        import PIL.Image
        PIL.Image.fromarray(rgb).save(out_dir / f"{prefix}_rgb.png")
    except Exception:
        import imageio
        imageio.imwrite(out_dir / f"{prefix}_rgb.png", rgb)

    # partseg：存为单通道 PNG，像素值即 0/1/2/...（训练可直接读）
    mask_u8 = np.clip(part_mask, 0, 255).astype(np.uint8)
    try:
        import PIL.Image
        PIL.Image.fromarray(mask_u8).save(out_dir / f"{prefix}_partseg.png")
    except Exception:
        import imageio
        imageio.imwrite(out_dir / f"{prefix}_partseg.png", mask_u8)

    if depth is not None:
        np.save(out_dir / f"{prefix}_depth.npy", depth)

    with (out_dir / f"{prefix}_meta.json").open("w", encoding="utf-8") as f:
        json.dump(view_meta, f, indent=2, ensure_ascii=False)
    log.debug("已写入 %s %s", out_dir, prefix)


# -----------------------------------------------------------------------------
# 单 object 完整渲染流程
# -----------------------------------------------------------------------------


def render_one_object(
    object_meta_path: Path,
    object_meta: Dict[str, Any],
    views: List[Tuple[float, float]],
    renders_root: Path,
    image_width: int,
    image_height: int,
    render_depth: bool,
    copy_object_meta: bool,
    log: logging.Logger,
    debug_mode: bool = False,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    渲染一个 object 的所有指定视角；写入 renders_root/<category>/<model_id>/。
    返回 (成功与否, 错误信息, 本 object 统计)。
    """
    object_id = object_meta.get("object_id", "unknown")
    category = object_meta.get("category", "unknown")
    model_id = object_meta.get("model_id", "unknown")
    part_order = object_meta.get("part_order", [])
    # 备用数据目录：project/data/<category>/<model_id>，与 scan_dataset 的 data 结构一致
    processed_root = renders_root.parent
    data_root = processed_root.parent / "data"
    fallback_source_dir = data_root / category / model_id

    # debug：打印 part 与 mesh / bbox 信息
    if debug_mode:
        part_ids = [p.get("part_id") for p in object_meta.get("parts", [])]
        log.info("object %s part_id 列表: %s", object_id, part_ids)
        for p in object_meta.get("parts", []):
            pid = p.get("part_id")
            meshes = p.get("mesh_files") or []
            plys = p.get("ply_files") or []
            bbox = p.get("bbox") or {}
            log.info(
                "part_id=%s mesh_files=%d ply_files=%d bbox_center=%s bbox_size=%s",
                pid,
                len(meshes),
                len(plys),
                bbox.get("center"),
                bbox.get("size"),
            )

    part_meshes = load_part_meshes(object_meta, log, fallback_source_dir=fallback_source_dir)
    if not part_meshes:
        return False, "无有效 part mesh", {"n_parts": 0, "n_views_ok": 0, "n_views": len(views)}

    center, size = compute_overall_bbox(part_meshes)
    if debug_mode:
        log.info("overall bbox center=%s size=%s", center.tolist(), size.tolist())
    distance = float(np.max(size) * BBOX_DISTANCE_FACTOR + BBOX_DISTANCE_MARGIN)

    # 将各 part 的 mesh 平移到以原点为中心（相机看原点）
    part_meshes_centered: List[Tuple[int, trimesh.Trimesh]] = []
    for part_id, mesh in part_meshes:
        m2 = mesh.copy()
        m2.vertices -= center
        part_meshes_centered.append((part_id, m2))

    out_dir = renders_root / category / model_id
    n_views_ok = 0
    view_stats: List[Dict[str, Any]] = []

    for view_idx, (az_deg, el_deg) in enumerate(views):
        try:
            rgb, part_mask, depth, _ = render_single_view(
                part_meshes_centered,
                np.zeros(3),
                az_deg,
                el_deg,
                distance,
                image_width,
                image_height,
                part_order,
                render_depth,
                log,
            )
            visible_parts, part_bboxes_2d = compute_visible_parts_and_2d_bboxes(part_mask)
            mask_encoding = {"0": "background"}
            for pid in part_order:
                mask_encoding[str(pid + 1)] = f"part_{pid}"
            view_meta = {
                "object_id": object_id,
                "category": category,
                "model_id": model_id,
                "view_id": view_idx,
                "camera": {
                    "azimuth_deg": az_deg,
                    "elevation_deg": el_deg,
                    "distance": distance,
                    "look_at": [0.0, 0.0, 0.0],
                },
                "image_size": [image_height, image_width],
                "visible_parts": visible_parts,
                "part_bboxes_2d": part_bboxes_2d,
                "mask_encoding": mask_encoding,
            }
            save_view_outputs(out_dir, view_idx, rgb, part_mask, depth, view_meta, log)
            # 保存原始 mask 数组，便于离线 debug
            if debug_mode:
                np.save(out_dir / f"view_{view_idx:03d}_partseg_raw.npy", part_mask)
                log.debug("view_%03d mask unique: %s", view_idx, np.unique(part_mask))
            n_views_ok += 1
            view_stats.append({"view_id": view_idx, "ok": True, "visible_parts": len(visible_parts)})
        except Exception as e:
            log.warning("视角 view_%s 渲染失败 %s: %s", view_idx, object_id, e)
            view_stats.append({"view_id": view_idx, "ok": False, "error": str(e)})

    if copy_object_meta:
        out_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        dst = out_dir / "object_meta.json"
        shutil.copy2(object_meta_path, dst)
        log.debug("已复制 object_meta.json 到 %s", dst)

    stats = {
        "n_parts": len(part_meshes),
        "n_views": len(views),
        "n_views_ok": n_views_ok,
        "view_stats": view_stats,
    }
    return n_views_ok > 0, "", stats


# -----------------------------------------------------------------------------
# 主流程：索引过滤、小批量、日志与统计
# -----------------------------------------------------------------------------


def setup_logging(logs_dir: Path) -> logging.Logger:
    """配置渲染日志：文件 + 控制台。"""
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "render_part_masks.log"
    logger = logging.getLogger("render_part_masks")
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
    parser = argparse.ArgumentParser(description="离屏渲染 object_meta 多视角 RGB / part mask / depth")
    parser.add_argument("--processed_root", type=str, default=str(paths["processed_root"]))
    parser.add_argument("--index", type=str, default=str(paths["object_meta_index"]))
    parser.add_argument("--renders_root", type=str, default=str(paths["renders_root"]))
    parser.add_argument("--categories", type=str, nargs="*", default=None, help="只渲染这些类别，默认全部 7 类")
    parser.add_argument("--limit_per_category", type=int, default=1, help="每类最多渲染几个 object（小批量验证默认 1）")
    parser.add_argument("--num_views", type=int, default=3, help="每个 object 渲染几个视角（小批量默认 3）")
    parser.add_argument("--debug_single_object", type=str, default=None, help="只渲染指定 object_id 的一个 object")
    parser.add_argument("--dry_run", action="store_true", help="只打印将要渲染的 object/视角，不写文件")
    parser.add_argument("--width", type=int, default=DEFAULT_IMAGE_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_IMAGE_HEIGHT)
    parser.add_argument("--no_depth", action="store_true", help="不渲染 depth")
    parser.add_argument("--no_copy_meta", action="store_true", help="不复制 object_meta.json 到渲染目录")
    args = parser.parse_args()

    processed_root = Path(args.processed_root).expanduser().resolve()
    index_path = Path(args.index).expanduser().resolve()
    renders_root = Path(args.renders_root).expanduser().resolve()
    logs_dir = processed_root / "logs"
    stats_dir = processed_root / "stats"
    debug_dir = processed_root / "debug_renders"

    log = setup_logging(logs_dir)
    log.info("渲染脚本启动；index=%s", index_path)

    if not index_path.is_file():
        log.error("object_meta_index 不存在: %s", index_path)
        sys.exit(1)

    with index_path.open("r", encoding="utf-8") as f:
        index_list = json.load(f)

    from collections import defaultdict
    categories = args.categories or DEFAULT_CATEGORIES
    index_list = [e for e in index_list if e.get("category") in categories]
    if args.debug_single_object:
        index_list = [e for e in index_list if e.get("object_id") == args.debug_single_object]
        if not index_list:
            log.error("未找到 object_id=%s", args.debug_single_object)
            sys.exit(1)
    else:
        per_cat = defaultdict(int)
        filtered = []
        for e in index_list:
            cat = e.get("category", "")
            if per_cat[cat] >= args.limit_per_category:
                continue
            filtered.append(e)
            per_cat[cat] += 1
        index_list = filtered

    views = generate_camera_views(
        DEFAULT_AZIMUTH_DEG,
        DEFAULT_ELEVATION_DEG,
        num_views=args.num_views,
    )
    log.info("待渲染 object 数: %s，每 object 视角数: %s", len(index_list), len(views))
    if args.dry_run:
        for e in index_list:
            log.info("[dry_run] %s", e.get("object_id"))
        log.info("dry_run 结束")
        return

    success_count = 0
    fail_count = 0
    errors: List[Dict[str, Any]] = []
    per_cat_count: Dict[str, int] = defaultdict(int)
    per_cat_visible: Dict[str, List[int]] = defaultdict(list)
    per_object_view_ok: List[float] = []

    for i, entry in enumerate(index_list):
        object_id = entry.get("object_id", "?")
        object_meta_path = Path(entry.get("object_meta_path", ""))
        log.info("[%s/%s] 渲染 %s", i + 1, len(index_list), object_id)
        try:
            object_meta = load_object_meta(object_meta_path)
            ok, err_msg, obj_stats = render_one_object(
                object_meta_path,
                object_meta,
                views,
                renders_root,
                args.width,
                args.height,
                render_depth=not args.no_depth,
                copy_object_meta=not args.no_copy_meta,
                log=log,
                debug_mode=bool(args.debug_single_object),
            )
            if ok:
                success_count += 1
                cat = object_meta.get("category", "unknown")
                per_cat_count[cat] += 1
                for vs in obj_stats.get("view_stats", []):
                    if vs.get("ok"):
                        per_cat_visible[cat].append(vs.get("visible_parts", 0))
                n_ok = obj_stats.get("n_views_ok", 0)
                n_tot = obj_stats.get("n_views", 1)
                per_object_view_ok.append(n_ok / n_tot)
            else:
                fail_count += 1
                errors.append({"object_id": object_id, "error": err_msg or "无有效 part mesh"})
        except Exception as e:
            log.exception("渲染失败 %s: %s", object_id, e)
            fail_count += 1
            errors.append({"object_id": object_id, "error": str(e)})

    # 统计写入
    stats = {
        "n_success": success_count,
        "n_fail": fail_count,
        "per_category_count": dict(per_cat_count),
        "per_category_avg_visible_parts": {
            cat: round(sum(per_cat_visible[cat]) / len(per_cat_visible[cat]), 2) if per_cat_visible[cat] else 0
            for cat in per_cat_count
        },
        "avg_view_success_rate_per_object": round(sum(per_object_view_ok) / len(per_object_view_ok), 4) if per_object_view_ok else 0,
        "errors": errors,
    }
    stats_dir.mkdir(parents=True, exist_ok=True)
    with (stats_dir / "render_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    log.info("渲染结束：成功 %s，失败 %s；统计已写 %s", success_count, fail_count, stats_dir / "render_stats.json")

    # 小批量时把每类已渲染的 object 复制到 debug_renders，便于人工检查
    if success_count > 0 and args.limit_per_category <= 2 and not args.debug_single_object:
        import shutil
        debug_dir.mkdir(parents=True, exist_ok=True)
        for e in index_list:
            cat = e.get("category", "")
            model_id = e.get("model_id", "")
            src = renders_root / cat / model_id
            if src.is_dir():
                dst = debug_dir / f"{cat}_{model_id}"
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
                log.info("已复制到 debug_renders: %s", dst.name)


if __name__ == "__main__":
    main()
