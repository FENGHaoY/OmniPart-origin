# 基于现成 40 张图 + SAM/DINO + object 结构约束的伪标签 Pipeline 设计

## 一、目标

为每个 object 的 **现成 40 张图像** 自动生成 **part-level pseudo segmentation mask**，供后续 part segmentation 模块训练使用。不依赖重渲染、不恢复相机，直接利用数据集中已有的 `imgs/` 与 `object.json` / `object_meta.json` 结构先验。

## 二、数据背景

- 数据根目录：`../project/data`，7 类，每类下 `<model_id>/` 为单 object。
- 每 object 目录含：`imgs/`（**40 张图**，如 `00.png`～`39.png`）、`object.json`、`objs/`、`plys/` 等。
- 已有索引：`dataset_index_all.json`（object 列表）、`object_meta_index.json`（object_meta 路径），以及每 object 的 `object_meta.json`（part 数、part_order、parent-child、articulated/fixed 等）。

## 三、Pipeline 如何工作

1. **Image index**：对每个 object 扫描 `imgs/`，建立稳定 `view_id`（0～39）与图像路径的对应，供后续“每 object 40 张图”的批量处理。
2. **SAM proposals**：对每张图运行 SAM（或兼容方案），得到候选区域 masks，保存原始 proposal。  
   **掩码约定**：SAM 只输出前景区域，无“背景掩码”；proposal 下标 0 表示第 1 个区域、1 表示第 2 个区域；背景 = 所有 proposal 在该像素均为 0。可视化图中未覆盖像素保持原图混合效果，无固定“背景色”。
3. **Region features**：对每张图的每个候选区域用 DINO/DINOv2 提取 pooled 特征。
4. **结构约束**：从 `object_meta.json` 读取 part 数、part_order、articulated/fixed、parent-child 等，作为 object-level 先验。
5. **Pseudo label 生成**：在“part 数量约束 + 结构先验”下，对 SAM 区域做筛选、合并、聚类与 region-to-part 对齐，得到每张图的 part-level pseudo mask 与 meta。
6. **训练索引**：汇总所有 pseudo masks，生成供 segmentation 训练使用的 dataset index。

目标不是完美 GT，而是**结构合理、可解释、可训练**的伪标签。

## 四、如何利用每个 object 的 40 张图

- 每条样本固定 **40 个 view**，`view_id` 与 `imgs/` 下文件名顺序一致（如 `00.png`→0，…，`39.png`→39），保证可复现。
- 后续 SAM、特征、伪标签均按 `(object_id, view_id)` 组织；统计与索引中显式记录每 object 的 view 数（目标 40，缺图时可少于 40）。

## 五、如何利用 object-level 结构约束

- **Part 数量**：object 的 logical part 数来自 `object_meta.parts`，最终 pseudo parts 数量应与之接近，避免无限碎片化。
- **Articulated / fixed**：可动 part 优先保持独立区域；fixed 小零件可启发式合并到父部件。
- **Parent-child**：合并/归并时作为先验（如 handle 更可能归到 drawer 而非 base）。
- **Part name**：可选语义提示，第一版可作预留接口。

这些约束在 `generate_pseudo_part_labels.py` 中显式参与：筛选、合并、聚类与 region-to-part 分配，并在注释中说明每条规则的作用与可替换为学习方法的点。

## 六、小批量验证

- 默认：每类 1 个 object、每 object 3～5 张图；通过人工检查后再扩展到 40 张/全量。
- 参数：`--categories`、`--limit_per_category`、`--limit_views_per_object`、`--debug_single_object`、`--dry_run`。
- 先跑通 `build_image_index_from_dataset.py`，确认 40 张图索引正确，再逐步实现 SAM、特征、伪标签与索引脚本。

## 七、输出目录（pseudo_seg）

- `proposals/`、`region_features/`、`pseudo_masks/` 按 `<category>/<model_id>/` 组织。
- `indexes/`：image_index、最终 `pseudo_seg_dataset_index.json`。
- `logs/`、`stats/`：日志与统计，便于追溯与对比。

## 八、已实现脚本与用法

1. **build_image_index_from_dataset.py**：扫描 data 的 imgs/，写 `indexes/image_index.json`。
2. **run_sam_proposals.py**：按 image_index 对每张图跑 SAM，写 `proposals/<cat>/<mid>/view_xxx_sam_masks.json|.npz`、`view_xxx_sam_vis.png`。
3. **extract_region_features.py**：对每张图的每个 SAM 区域用 DINOv2 做 masked mean pool，写 `region_features/<cat>/<mid>/view_xxx_region_features.npz|.json`。
4. **generate_pseudo_part_labels.py**：结合 proposals、region features、object_meta（part 数、part_order），K-means 聚类 region→part，写 `pseudo_masks/<cat>/<mid>/view_xxx_pseudo_partseg.png`、`view_xxx_pseudo_meta.json`。
5. **build_pseudo_seg_dataset_index.py**：汇总有 pseudo_partseg 的 (object, view)，写 `indexes/pseudo_seg_dataset_index.json` 供训练使用。

小批量参数（各脚本通用）：`--categories`、`--limit_per_category`、`--limit_views_per_object`、`--debug_single_object`、`--dry_run`。全量时去掉 limit 即可。
