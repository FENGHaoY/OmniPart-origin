# render_part_masks_pyrender 设计说明

## 目标

读取标准化 `object_meta.json`，用 trimesh + pyrender 在 aarch64 离屏渲染每个 object 的多视角：RGB、part segmentation mask、depth（可选）、每视角 meta。用于部件级分割训练与 joint-aware bboxgen 的 image/mask 条件。

## 渲染流程概览

1. 加载 `object_meta_index.json`，按 `--categories` / `--limit_per_category` 过滤得到待渲染 object 列表。
2. 对每个 object：
   - `load_object_meta(object_meta_path)` 得到 object_meta。
   - 解析 `source_object_dir`，作为 mesh 根目录。
   - `load_part_meshes(object_meta)`：对每个 part，优先用 `mesh_files`（否则 `ply_files`），从 `source_object_dir` 下加载并合并为单个 trimesh，得到 `[(part_id, trimesh), ...]`；无有效 mesh 的 part 跳过并记日志。
   - `compute_overall_bbox(all_part_meshes)`：合并所有 part 的顶点，计算整体 AABB，得到 center 与 size（或 min/max）。
   - `generate_camera_views(...)`：按配置生成视角列表（默认前半球 21 个；小批量时用 `--num_views` 取前 N 个）。
   - 根据 overall bbox 计算相机距离：例如 `distance = max(size) * scale + margin`，保证物体完整入画。
   - 对每个视角调用 `render_single_view`：构建场景、按 part 上色、渲染 RGB 与 mask（每 part 唯一颜色），再 `decode_part_mask` 得到 0/1..N 的整型 mask；可选渲染 depth。
   - `compute_visible_parts_and_2d_bboxes(mask)`：从 mask 统计 visible_parts，并对每个 part 算 2D bbox。
   - `save_view_outputs`：写出 view_xxx_rgb.png、view_xxx_partseg.png、view_xxx_depth.npy（可选）、view_xxx_meta.json；并复制或链接 object_meta.json 到渲染目录。
3. 汇总日志与 `render_stats.json`，失败 object 记录原因，不中断整体。

## 前半球视角设计

- **为何用前半球**：物体大多有 canonical front，articulation 主要在正面；背面对 part segmentation 意义小且易引入噪声。
- **为何不用全 360°**：减少无效视角、节省算力，且与下游“正面为主”的设定一致。
- **第一版模板**：
  - azimuth（度）：[-75, -50, -25, 0, 25, 50, 75]（7 个）
  - elevation（度）：[15, 35, 55]（3 个）
  - 共 7×3 = 21 个视角。
- **实现**：`generate_camera_views(azimuth_deg_list, elevation_deg_list)` 返回 `[(azimuth, elevation), ...]`；脚本内默认传入上述列表，小批量时只取前 `num_views` 个。
- **后续调整**：更密/更疏只需改传入的 azimuth_deg_list / elevation_deg_list 或对返回列表做采样。

## Mask 解码方案

- **编码规则**：0 = 背景；1..N = part_id + 1（避免 part_id=0 与背景混用）。
- **实现**：每个逻辑 part 在渲染时赋予**唯一纯色**（如 (part_id+1)/255 或按 (R,G,B) 离散分配保证可逆）；渲染一次 color pass，得到 (H,W,3) 或 (H,W,4)。解码时：对每个像素，若与某 part 颜色匹配（允许小误差）则 mask[i,j]=part_id+1，否则 0。为稳定，part 颜色用整数 RGB 并归一化到 0–1 给 pyrender，解码时四舍五入或容差比较。
- **同一逻辑 part 的多个 mesh**：合并为一个 trimesh 后只赋一种颜色，故天然同一 part 一个 id。

## 相机距离与归一化

- 用所有 part 合并后的顶点计算 overall bbox（center + size）。
- 设 `d = max(size) * factor + margin`（如 factor=1.2, margin=0.3），相机距离 = d，look_at = bbox center；物体中心可先平移到原点再渲染，或 look_at 设为 center。保证不同尺度 object 都能完整出现在画面内、不被裁切。
- 参数集中为常量或命令行可调，避免魔法数字。

## 小批量验证方案

- **目标**：每类先渲染 1～2 个 object，每个 object 先 3～5 个视角，人工检查后再全量。
- **参数**：
  - `--limit_per_category`：每类最多渲染几个 object（默认 1）。
  - `--num_views`：每个 object 渲染几个视角（默认 3，从 21 个中取前 3 个）。
  - `--categories`：只渲染这些类别，默认全部 7 类。
  - `--debug_single_object`：只渲染指定 object_id 的一个 object。
  - `--dry_run`：只打印将要渲染的 object 和视角，不写文件。
- **默认验证模式**：limit_per_category=1, num_views=3；通过后再改为 2/21 或全量。

## 输出与可追溯

- 渲染目录：`renders/<category>/<model_id>/`，内含 view_xxx_*.png/.npy/.json 及 object_meta.json。
- 日志：`logs/render_part_masks.log`。
- 统计：`stats/render_stats.json`（成功/失败数、每类数量、每类平均可见 part、每 object 平均视角成功率等）。
- 失败记录：每个失败 object 记录原因（如 mesh 缺失、pyrender 报错等），不崩溃。
