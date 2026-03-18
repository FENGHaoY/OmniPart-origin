# 伪标签 Part Segmentation Pipeline 说明

基于现成 40 张图 + SAM + DINOv2 + object_meta 结构约束，为每个 object 的每张图生成 **part-level pseudo segmentation mask**，供后续 part segmentation 模块训练。设计见 `research/docs/design_pseudo_seg_pipeline.md`。

## 依赖与路径

- **数据**：`../project/data`（7 类），`../project/processed_data/`（索引、object_meta、proposals、region_features、pseudo_masks）。
- **索引**：`dataset_index_all.json`、`indexes/object_meta_index.json`、`pseudo_seg/indexes/image_index.json`。
- **Python**：torch、transformers（DINOv2）、segment_anything（SAM）、scikit-learn（KMeans）、numpy、PIL/cv2。

## Pipeline 顺序

1. **build_image_index_from_dataset.py**  
   扫描 data 的 imgs/，生成 `pseudo_seg/indexes/image_index.json`（每 object 的 view 列表与图像路径）。

2. **run_sam_proposals.py**  
   对 image_index 中每张图跑 SAM，写出 `pseudo_seg/proposals/<category>/<model_id>/view_xxx_sam_masks.json|.npz`、`view_xxx_sam_vis.png`。

3. **extract_region_features.py**  
   对每张图加载 DINOv2，按每个 SAM 区域做 patch 特征 masked mean pool，写出 `pseudo_seg/region_features/<category>/<model_id>/view_xxx_region_features.npz|.json`。

4. **generate_pseudo_part_labels.py**  
   读取 proposals、region_features、`objects_meta/<category>/<model_id>/object_meta.json`；按 part 数 K-means 聚类区域并生成 part mask，写出 `pseudo_seg/pseudo_masks/<category>/<model_id>/view_xxx_pseudo_partseg.png`、`view_xxx_pseudo_meta.json`。

5. **build_pseudo_seg_dataset_index.py**  
   汇总所有存在 pseudo_partseg 的 (object, view)，写出 `pseudo_seg/indexes/pseudo_seg_dataset_index.json`，供训练脚本按样本读取 image_path、pseudo_partseg_path 等。

## 小批量验证

各脚本支持统一参数（与 design 一致）：

- `--limit_per_category N`：每类最多 N 个 object  
- `--limit_views_per_object N`：每 object 最多 N 张图  
- `--debug_single_object <category>_<model_id>`：只跑指定 object  
- `--dry_run`：只打日志不写文件  

示例（每类 1 个 object、每 object 3 张图）：

```bash
cd research
python build_image_index_from_dataset.py --limit_per_category 1 --limit_views_per_object 3
python run_sam_proposals.py --limit_per_category 1 --limit_views_per_object 3
python extract_region_features.py --limit_per_category 1 --limit_views_per_object 3
python generate_pseudo_part_labels.py --limit_per_category 1 --limit_views_per_object 3
python build_pseudo_seg_dataset_index.py --limit_per_category 1 --limit_views_per_object 3
```

全量：去掉 `--limit_per_category` 与 `--limit_views_per_object` 即可（需先全量跑过 build_image_index_from_dataset.py）。

## 输出约定

- **Part mask 编码**：`view_xxx_pseudo_partseg.png` 单通道 uint8：0=背景，1..K=part（对应 part_order 的 1-based 下标）。
- **pseudo_seg_dataset_index.json**：`samples[]` 每项含 `object_id`、`category`、`model_id`、`view_id`、`image_path`、`pseudo_partseg_path`、`pseudo_meta_path`、`n_parts`、`part_order`。

## 限制与后续可改进

- 伪标签质量依赖 SAM 覆盖与 DINOv2 特征、以及 K-means 与 part 数对齐，无 GT 监督。
- 第一版未用 object_meta 的 articulated/fixed、parent-child 做合并/归并规则，仅用 part 数量约束；后续可在 generate_pseudo_part_labels.py 中增加启发式或学习式分配。
