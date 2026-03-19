# CAP-net 预处理：构造 512×512 训练数据

这套预处理用于把 CAP-net 的渲染图 + 分割掩码（`.npz`）变成更适合训练的 512×512 样本，核心目标是：

- **减少背景占比**（裁剪前景 bbox）
- **尽量不形变**（保持长宽比缩放）
- **背景统一**（用掩码反向把背景像素统一填灰，消除“原背景 vs padding”边界感）
- **掩码不被插值污染**（mask resize 一律用最近邻 / `order=0`）

相关代码集中在：

- `research/capnet_preprocess/preprocess_to_512.py`
- `research/capnet_preprocess/analyze_segmentation_npz.py`

---

## 需要的源数据

### 批量处理（目录）
`--input_root` 指向 CAP-net 的 `sample_data` 根目录，至少包含：

- `rgb/`：渲染图（`.png`/`.jpg`）
- `segmentation/`：与 `rgb` 同 stem 的 `.npz`

每个 `.npz` 需要包含（2D，shape 与 RGB 相同）：

- `semantic_segmentation`：`int32`（约定 **-2 = 背景**）
- `instance_segmentation`：`int32`
- `depth_segmentation`：`bool`

### 语义编号约定（当前 CAP-net 数据）

你已确认的语义映射如下（来自 `metafile/*.json` 的 `target_gaparts`）：

- `0`: `line_fixed_handle`
- `1`: `round_fixed_handle`
- `2`: `slider_button`
- `3`: `hinge_door`
- `4`: `slider_drawer`
- `5`: `slider_lid`
- `6`: `hinge_lid`
- `7`: `hinge_knob`
- `8`: `hinge_handle`

特殊值约定：

- `-2`: 背景（background）
- `-1`: base

### 单条处理（单文件）
直接传 `--rgb` + `--npz`。

---

## 输出结构

`--out_root` 下会生成：

- `rgb_512/*.png`
- `segmentation_512/*.npz`
- `debug_vis/`（可选）：
  - `*_rgb.png`
  - `*_semantic_color.png` / `*_semantic_overlay.png`
  - `*_instance_color.png` / `*_instance_overlay.png`
  - `*_semantic_*_legend.png` / `*_instance_*_legend.png`（右侧附带颜色→label 标注）
- `preprocess_report.json`（批量）或 `preprocess_report_single.json`（单条）

---

## 推荐策略（默认就按这个）

### `fit_pad_bg` + `bg_fill=gray`

这是当前最适合“训练数据构造”的设置：

- **先裁剪 bbox**（减少背景）
- **保持长宽比缩放**到画布内部（减少形变）
- **padding + 原背景都统一填灰**（用 `semantic=-2` 反向覆盖背景像素）

关键参数：

- `--margin`：bbox 外扩像素（越大背景越多）
- `--fill_fraction`：内容在 512×512 中占比（越小背景越多、形变越小）
- `--bg_gray`：背景灰度值（推荐 140–180）

常见调参建议：

- 想更像 `singapo` 那种“背景占比明显但不空”：先把 `--fill_fraction` 调到 `0.65~0.8`，再根据需要调 `--margin`（如 20/30/40）。
- 如果你发现物体仍然偏大（接近顶满画面）：减小 `--fill_fraction` 或增大 `--margin`。
- 如果你觉得背景太多、物体偏小：增大 `--fill_fraction` 或减小 `--margin`。

---

## 运行命令

### 1) 处理一条数据（单张）

```bash
conda activate omnipart
python research/capnet_preprocess/preprocess_to_512.py \
  --rgb /home/bingxing2/home/scx8q10/xiaoqian/project/sample_data/rgb/StorageFurniture_46179_0_0.png \
  --npz /home/bingxing2/home/scx8q10/xiaoqian/project/sample_data/segmentation/StorageFurniture_46179_0_0.npz \
  --out_root research/outputs/capnet_one_512 \
  --canvas 512 \
  --margin 30 \
  --output_mode fit_pad_bg \
  --fill_fraction 0.75 \
  --bg_fill gray \
  --bg_gray 160 \
  --debug_vis
```

生成的检查图在：

- `research/outputs/capnet_one_512/debug_vis/*_semantic_overlay.png`

### 2) 处理一个目录（批量）

```bash
conda activate omnipart
python research/capnet_preprocess/preprocess_to_512.py \
  --input_root /home/bingxing2/home/scx8q10/xiaoqian/project/sample_data \
  --out_root research/outputs/capnet_all_512 \
  --canvas 512 \
  --margin 30 \
  --output_mode fit_pad_bg \
  --fill_fraction 0.75 \
  --bg_fill gray \
  --bg_gray 160 \
  --debug_first_vis
```

批量模式只会为“第一张成功样本”写 debug 图，避免输出太多；完整统计看：

- `research/outputs/capnet_all_512/preprocess_report.json`

---

## 掩码可视化/检查（可选）

用于快速检查 `.npz` 的 label 分布/legend/overlay：

```bash
conda activate omnipart
python research/capnet_preprocess/analyze_segmentation_npz.py \
  --rgb /home/bingxing2/home/scx8q10/xiaoqian/project/sample_data/rgb/StorageFurniture_46179_0_0.png \
  --npz /home/bingxing2/home/scx8q10/xiaoqian/project/sample_data/segmentation/StorageFurniture_46179_0_0.npz \
  --out_dir research/outputs/capnet_mask_debug/view0
```

