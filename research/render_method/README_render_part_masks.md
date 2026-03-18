# render_part_masks_pyrender 说明

## 依赖

- **trimesh**：已在 OmniPart requirements 中。
- **pyrender**：需单独安装，`pip install pyrender`。  
  在无显示器服务器（如 aarch64 头节点）上，脚本会自动设置 `PYOPENGL_PLATFORM=egl`；若仍报错，可先执行 `export PYOPENGL_PLATFORM=egl` 再运行脚本，或安装 OSMesa 等软件渲染后端。

## 1. 脚本解决什么问题

读取已标准化的 `object_meta.json`，用 **trimesh + pyrender** 在 aarch64 离屏渲染每个 object 的多视角：
- **RGB 图**：用于可视化和后续 image 条件；
- **part segmentation mask**：按逻辑 part 编码，用于部件级分割训练和 bboxgen 的 mask 条件；
- **depth**（可选）：用于 3D 相关分析；
- **每视角 meta**：相机参数、visible_parts、2D bbox、mask 编码，便于训练与调试。

## 2. 输入

- **object_meta_index.json**：路径默认 `../project/processed_data/indexes/object_meta_index.json`，列出所有待渲染 object 的 `object_meta_path`。
- **每个 object 的 object_meta.json**：由 `parse_object_json.py` 生成，含 `source_object_dir`、`parts`（含 `mesh_files`/`ply_files`）。
- **Mesh 文件**：位于各 object 的 `source_object_dir` 下，优先使用 `mesh_files`，为空时使用 `ply_files`。

## 3. 输出

- **渲染目录**：`../project/processed_data/renders/<category>/<model_id>/`
  - `view_000_rgb.png`、`view_000_partseg.png`、`view_000_depth.npy`（可选）、`view_000_meta.json`
  - 可选复制 `object_meta.json` 到同目录，便于自描述。
- **日志**：`../project/processed_data/logs/render_part_masks.log`
- **统计**：`../project/processed_data/stats/render_stats.json`（成功/失败数、每类数量、平均可见 part、视角成功率等）

## 4. 为什么按逻辑 part 渲染，而不是按单个 mesh 渲染

一个逻辑 part 在数据中可能对应**多个 mesh 文件**（如一个门由多块几何组成）。若按单 mesh 渲染，同一逻辑部件会被拆成多个 mask id，与 part 级标注和 joint 结构不一致。因此按 **object_meta 的 parts[]** 为单位：每个 part 的所有 mesh 合并为一个 trimesh，渲染时赋**同一个 mask id**，保证 part segmentation 与后续 joint token 对齐。

## 5. Mask 编码规则

- **0** = 背景  
- **1..N** = part_id + 1（即 part_0 对应 1，part_1 对应 2，…）

这样避免 part_id=0 与背景共用 0。  
每视角的 `view_xxx_meta.json` 中会写出 `mask_encoding`，例如 `"1": "part_0"`，便于自解释和训练时映射。

## 6. 为什么采用前半球视角，而不是全 360°

- 物体大多有 **canonical front**，articulation（门、抽屉等）主要在正面可见。
- 背面视角对 part segmentation 训练意义小，且易引入遮挡和噪声。
- 前半球可减少无效视角、节省算力，与下游“正面为主”的设定一致。  
第一版模板：azimuth [-75,-50,-25,0,25,50,75]（度），elevation [15,35,55]（度），共 21 个视角。要更密/更疏只需改脚本内 `DEFAULT_AZIMUTH_DEG` / `DEFAULT_ELEVATION_DEG` 或 `--num_views`。

## 7. 2D bbox 和 visible_parts 怎么从 mask 算出来

- **visible_parts**：对当前视角的 part mask 做 `np.unique`，去掉 0（背景），得到出现过的 part_id 列表（0-based），即该视角可见的 part。
- **part_bboxes_2d**：对每个在 mask 中出现的 part_id，取该 part 所有像素的 x、y 最小/最大值，得到 `[xmin, ymin, xmax, ymax]`（整数像素坐标），写入 `view_xxx_meta.json` 的 `part_bboxes_2d`，key 为 part_id 字符串。

## 8. 如何先小批量验证，再扩展到大规模渲染

- **小批量验证**（默认）：`--limit_per_category 1`、`--num_views 3`，即每类只渲染 1 个 object、每个 object 只渲染 3 个视角。用于确认 pipeline、视角、mask、2D bbox 是否正确。
- **单 object 调试**：`--debug_single_object Dishwasher_11622` 只渲染指定 object_id。
- ** dry_run**：`--dry_run` 只打印将要渲染的 object 和视角数，不写文件。
- **扩展**：确认无误后，增大 `--limit_per_category`、`--num_views 21` 或去掉 limit 做全量渲染（需在代码中支持“不限制每类数量”时传大数或单独逻辑）。

## 9. 后续使用这些输出的模块

- **部件级分割模块训练**：用 RGB + partseg.png（或从 meta 的 mask_encoding 解析）作为监督；
- **joint-aware bboxgen 的 image/mask 输入**：用 RGB 与 part mask 作为条件；
- **结果可视化与检查**：用 view_xxx_rgb.png、view_xxx_partseg.png 与 view_xxx_meta.json 做人工抽查和调试。
