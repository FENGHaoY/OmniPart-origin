# Joint-aware BBox Token 扩展计划

目标：将关节信息并入 `bboxgen` 的 token 表示，形成单阶段 `joint-aware bbox generation`。

---

## 1) 原 bbox token 字段（当前）

当前每个 part 仅包含 bbox 两个角点：

| 字段 | 语义 | 类型 | 编码方式 | tokens/part |
|---|---|---|---|---|
| `bbox_min_xyz` | 包围盒最小角点 | continuous | quantize 到 `bins=64` | 3 |
| `bbox_max_xyz` | 包围盒最大角点 | continuous | quantize 到 `bins=64` | 3 |

序列结构（概念）：

`[BOS] + part_1(6) + part_2(6) + ... + [EOS]`

说明：

- 词表：`0..63` 为空间离散坐标；`64/65/66` 为 `BOS/EOS/PAD`
- 语法约束在 `MeshDecodeLogitsProcessor` 中实现

---

## 2) 扩展后 token 字段建议（joint-aware）

### 2.1 推荐 v1（全 token 离散，优先实现）

| 字段 | 语义 | 类型 | 建议 | tokens/part |
|---|---|---|---|---|
| `bbox_min_xyz` | bbox 最小角点 | continuous | quantized discrete | 3 |
| `bbox_max_xyz` | bbox 最大角点 | continuous | quantized discrete | 3 |
| `joint_type` | 关节类型 | categorical | token id (`none/revolute/prismatic/fixed/...`) | 1 |
| `parent_idx` | 父节点索引 | discrete | `0..K`（`0=root`） | 1 |
| `axis_xyz` | 关节轴方向 | continuous | quantized discrete | 3 |
| `pivot_xyz` | 关节点位置 | continuous | quantized discrete | 3 |
| `motion_range` | 运动范围 | continuous | quantized discrete (min/max) | 2 |

合计：`16 tokens/part`。

### 2.2 备选 v2（token + 连续残差）

- 保留 v1 全离散 token 作为主输出
- 额外在 decoder hidden 上加 residual regression（同模型内）细化：
  - `axis_xyz_residual`
  - `pivot_xyz_residual`
  - `range_residual`

此方案不是外挂后处理头，可作为精度增强阶段。

---

## 3) token 对齐规则（必须严格）

1. 定义全工程唯一 part 排序函数 `order_fn`。
2. 对每个样本得到 permutation `P`（原 part -> token 序 part）。
3. 所有 part 级字段统一按 `P` 重排：
   - bbox
   - joint_type
   - parent/child graph
   - axis/pivot/range
4. parent 索引必须重映射到新序：
   - `parent_new = invP[parent_old]`
   - 无父节点统一编码为 `0`
5. token block 序严格对应 part 序：
   - block i 仅监督第 i 个 part

---

## 4) supervision 对齐规则（训练）

### 4.1 监督生成顺序

`raw annotations -> reorder by P -> remap parent_idx -> tokenize -> build input_ids/labels`

### 4.2 条件掩码（joint_type 相关）

- 当 `joint_type` 为 `none/fixed` 时：
  - `axis/pivot/range` 可置默认值
  - loss 用 mask 屏蔽这些 slot

### 4.3 损失建议

- 主损失：token CE（全部离散 slot）
- slot-weight：joint 字段可加更高权重
- 可选：v2 残差 L1/L2（仅对有效 joint）

---

## 5) 字段离散化/连续化建议

### 建议优先离散化

- `joint_type`（天然 categorical）
- `parent_idx`（图结构离散索引）
- `axis/pivot/range`（先量化，便于统一 AR 解码与约束）

### 可保留连续回归（第二阶段增强）

- `axis/pivot/range` 的 residual refinement
- 仍需以 token 对齐为前提，不能拆成独立后处理模块

---

## 6) 训练与推理需修改的文件列表

## 6.1 推理侧（必须）

- `modules/bbox_gen/utils/bbox_tokenizer.py`
  - 扩展为 joint-aware token 编解码
- `modules/bbox_gen/models/bbox_gen_models.py`
  - 将 `MeshDecodeLogitsProcessor` 扩展为 slot-aware 语法约束
- `modules/bbox_gen/models/autogressive_bbox_gen.py`
  - 更新 config（`vocab_size/max_length`）
  - `generate` 输出 joint-aware 结构
- `configs/bbox_gen.yaml`
  - `vocab_size/max_length/max_group_size` 等同步更新
- `scripts/inference_omnipart.py`
  - 导出 joint-aware 结果文件（不是只保存 `bboxes.npy`）
- `modules/inference_utils.py`
  - 增加 segmentation 输出适配（array -> ordered mask）可选入口

## 6.2 训练侧（必须）

当前仓库未公开 bboxgen 训练链，需新增：

- `research/` 或 `training/` 下新增 bboxgen dataset/collator
  - 负责 joint-aware sample -> `input_ids/labels`
- 新增 bboxgen trainer/script（teacher forcing + CE）
- 与 tokenizer 完全共用 schema，避免 train/infer 漂移

建议新增文件（命名可调整）：

- `research/training/jointaware_bbox_dataset.py`
- `research/training/jointaware_bbox_collator.py`
- `research/training/train_jointaware_bboxgen.py`
- `research/tokenization/jointaware_bbox_tokenizer.py`（或直接改原 tokenizer）

---

## 7) 长度预算与配置建议

若 `K_max = 50` 且 `tokens_per_part = 16`：

- 生成 token 预算约：`1(BOS) + 50*16 + 1(EOS) = 802`
- 当前 `voxel_token_length = 1886`
- 则 `max_length` 至少约 `1886 + 802 = 2688`

当前配置 `max_length=2187` 不足，需要提升并评估显存和训练吞吐。

---

## 8) 建议的样本格式（支持 joint-aware token supervision）

建议 `npz + json`：

```json
{
  "uuid": "...",
  "order_rule": "bottom_up_y_desc_37x37",
  "num_parts": 12,
  "part_name_ordered": ["base", "door", "handle", "..."],
  "quantization": {
    "coord_bins": 64,
    "axis_bins": 64,
    "pivot_bins": 64,
    "range_bins": 128
  }
}
```

```text
sample_xxx.npz
  image             [3,518,518] float32
  mask_raw          [H,W] int32
  ordered_mask_37   [37,37] int16
  voxel_coords      [N,3] int16
  bbox              [K,2,3] float32
  joint_type        [K] int8
  parent_idx        [K] int16
  axis              [K,3] float32
  pivot             [K,3] float32
  motion_range      [K,2] float32
  perm              [K] int16
  input_ids         [L] int32
  labels            [L] int32
```

---

## 9) 方案定位

- 主方案：joint-aware bbox token（一体化生成）
- baseline（可选对照）：bbox 后外挂 joint head（仅实验对照，不作为主路线）

