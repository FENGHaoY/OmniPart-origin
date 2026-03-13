# OmniPart 工程数据流梳理（Joint-aware BBox Token 视角）

本文严格围绕你的真实路线：

- 不是 `bbox -> 外挂 joint head`
- 而是 `joint information embedded into bbox token representation`

---

## Part I: OmniPart 原版 bboxgen 工程链梳理

### I.1 端到端推理链（当前仓库）

主入口是 `scripts/inference_omnipart.py`：

```text
(image, mask.exr)
  -> load_img_mask
  -> get_coords (part_synthesis pipeline)
  -> prepare_bbox_gen_input
  -> bbox_gen.generate
  -> bboxes.npy
  -> prepare_part_synthesis_input(voxel_coords + bboxes + mask)
  -> get_slat(..., part_layouts, masks)
```

关键中间文件：

- `voxel_coords.npy`: 稀疏结构坐标（阶段 0 输出）
- `bboxes.npy`: 阶段一 `bbox_gen` 输出
- `part_layouts`: 从 `voxel_coords + bboxes` 计算，驱动阶段二 part-wise 生成

---

### I.2 A. 部件级分割模块接入点（原版行为）

#### 1) 当前 mask 格式要求

- README 约定：mask 是 `.exr`，shape `[H, W, 3]`
- 三个通道存相同 `part_id`
- `scripts/inference_omnipart.py` 当前强制要求 `--mask_input`

当前读入路径：

- `modules/inference_utils.py::load_img_mask`
- `cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)` 读取 mask

#### 2) `ordered_mask_input` 的来源与用途

来源：

- `load_img_mask` 内部调用 `load_bottom_up_mask`
- `load_bottom_up_mask` 先把 mask 下采样到 `37x37`
- 再按“底到顶（y 最大值降序）”重排 part id，得到 `ordered_mask_input`（long tensor）

用途（推理时会被两处使用）：

- 阶段一 `bbox_gen`：作为 `GroupEmbedding` 输入（语义分组条件）
- 阶段二 `part_synthesis`：作为 `masks` 输入到 `structured_latent_flow`

#### 3) 你的 part segmentation module 最自然接入位置

最自然的是在 `scripts/inference_omnipart.py` 中 `load_img_mask(...)` 之前接入：

- 选项 A（最小侵入）：你的分割模块先产出 `mask.exr`（兼容现有格式），后续代码不改
- 选项 B（更干净）：改 `load_img_mask` 支持直接接收 `HxW` 或 `HxWx3` numpy mask，减少磁盘中转

建议先做 A，便于快速复用当前全链路。

#### 4) 为兼容 bboxgen，part ordering 需要满足什么要求

关键不是“分割模块输出标签值必须固定”，而是“最终进入 bboxgen 的 `ordered_mask_input` 必须稳定”。

当前规则是：

- 统一经 `load_bottom_up_mask` 重排
- part id 被重新映射为 `1..K`
- `0` 为背景

因此你的分割模块只需保证：

- 不同 part 可分离
- part 数不超过 `max_group_size`（当前 50）
- 细小 part 不应在 `37x37` 下采样中全部消失

---

### I.3 B. 关节参数预测接入点（先看原版 bboxgen）

#### 1) 当前 bboxgen 输入/输出/中间表示

输入（`prepare_bbox_gen_input`）：

- `points`: `[1, N, 3]`，连续坐标 `[-0.5, 0.5]`
- `whole_voxel_index`: `[1, N, 3]`，离散 `0..63`
- `images`: `[1, 3, 518, 518]`
- `masks`: `[1, 37, 37]`

中间表示（`autogressive_bbox_gen.py`）：

- 图像 token（DINOv2）
- mask group embedding 拼到图像 token
- 点云经 PartField + 3D encoder 得到 voxel token
- `voxel_token = image_token + voxel_token`
- decoder 自回归生成 bbox token 序列

输出：

- `bboxes`: `[K, 2, 3]`（decode 后连续坐标）

#### 2) 哪些文件负责 bbox token 定义/编码/解码/输出

- token 编解码：`modules/bbox_gen/utils/bbox_tokenizer.py`
- token 语法约束（生成时）：`modules/bbox_gen/models/bbox_gen_models.py` 中 `MeshDecodeLogitsProcessor`
- token 生成主逻辑：`modules/bbox_gen/models/autogressive_bbox_gen.py`
- token 容量配置：`configs/bbox_gen.yaml`

#### 3) 原版工程对训练的现实状态

仓库里有 `BboxGen.forward(batch)`，且 forward 依赖 `batch['input_ids']`，但当前公开代码未包含 bboxgen 完整训练入口（dataset/trainer/config）。

可见信息：

- 训练时应是“teacher forcing + token CE”风格（forward 返回 logits）
- `input_ids` 中使用 `voxel_token_placeholder=-1` 把条件 token 与文本 token拼在同一序列

这意味着 joint-aware 改造不仅要改模型，还要补齐/重建 bboxgen 数据与训练样本层。

---

### I.4 C. 训练样本层（原版可观察链路）

当前公开训练链路主要是阶段二（`training_part_synthesis`），其 `part_layouts + ordered_mask_dino` 已经是严格输入。

但 bboxgen 监督整理层（构造 bbox token target 的 dataset/collator）在公开代码中缺失。

因此：

- “bbox supervision 在哪一层整理”在当前仓库中不可直接追到实现文件
- 从 `BboxGen.forward` 可反推：它应发生在 bboxgen 专用 dataset/collator 里
- 你的 joint supervision 对齐，最佳落点也应放在同一层（tokenization 之前）

---

## Part II: 面向 joint-aware bbox token 扩展的改造建议

### II.1 总原则（与你路线一致）

目标模型应直接输出：

- `part bbox`
- `joint parameters`

并且两者共享同一套 part token ordering。

不建议把 joint 作为 bbox 后处理外挂头；那可以作为 baseline，但不是主方案。

---

### II.2 A. 分割模块接入改造建议

#### 推荐接入结构

```text
image
  -> your part segmentation module
  -> mask_pred (HxW int 或 HxWx3 float)
  -> load_bottom_up_mask-compatible adapter
  -> ordered_mask_input [37,37]
  -> bboxgen + part_synthesis
```

#### 建议新增薄适配层

在 `modules/inference_utils.py` 增加类似接口：

- `build_ordered_mask_from_array(mask_array)`
- 内部复用现在的 bottom-up 重排逻辑

这样 CLI 可支持：

- `--mask_input`（旧模式）
- `--auto_mask`（新模式，走你的分割模型）

#### ordering 约束（与你 C 点直接相关）

必须把“排序函数”定义成全工程唯一：

- 推理：用于 `ordered_mask_input`
- 训练：用于 bbox/joint target 排序

建议 v1 先沿用当前规则（2D y 降序），保证推理训练一致；后续再升级为更稳健的 canonical order。

---

### II.3 B. joint-aware bbox token 扩展建议

你要求的是“扩展 bbox token 本体”，因此需要同时改 5 层：

#### 1) token schema（必须改）

把每个 part 的 token block 从原 `6 tokens`（bbox）扩为 `6 + joint tokens`。

建议 v1 block：

- bbox_min/max: 6
- joint_type: 1
- parent_idx: 1
- axis: 3
- pivot: 3
- motion_range: 2

合计 `16 tokens/part`。

#### 2) dataset target（必须改）

构造 joint-aware target 时，必须先做 part 重排，再 tokenization。

核心是“先对齐后编码”：

1. 得到本样本 part permutation `P`
2. 对 `bbox/joint/graph` 同步重排
3. parent index 按 `P` 重映射
4. 统一编码成 token 序列

#### 3) model output head（轻改）

decoder 头仍可复用单一 LM head（分类到词表）。

需要改的是：

- `vocab_size`
- `max_length`
- 生成语法约束（不同 slot 的合法 token 范围）

#### 4) loss（必须改）

主损失仍是 token-level CE，但要加 slot mask：

- 对 `joint_type=none/fixed` 的 part，屏蔽不适用字段（axis/range 等）
- 可按字段做加权（joint 字段权重略高于 bbox 字段）

#### 5) inference export（必须改）

`generate` 输出不应只含 `bboxes`，应返回结构化对象：

- `bboxes`
- `joint_type`
- `parent_idx`
- `axis`
- `pivot`
- `motion_range`

并保存为 `jointaware_layout.npz/json`。

---

### II.4 字段离散化 vs 连续回归建议

在“token-first”前提下，推荐：

- 直接离散化（v1）：
  - `joint_type`（categorical）
  - `parent_idx`（discrete index）
  - `axis/pivot/range`（quantized discrete）
- 可选连续残差（v2）：
  - 在 decoder hidden 上加 residual regression（非外挂后处理）
  - 用于细化 axis/pivot/range

理由：

- 纯离散最容易并入现有 AR token 解码与语法约束
- 连续残差可做精度增强，但应保持“同一模型、同一 part 对齐”

---

### II.5 C. 训练样本层 joint supervision 接入建议

#### 1) joint supervision 最合适接入层

与 bbox target 同层：bboxgen dataset/collator 的 tokenization 前。

不要做独立 joint dataset 再后融合；会破坏 token 对齐链。

#### 2) 每个 part token 监督至少应包含

- `bbox_min[3], bbox_max[3]`
- `joint_type`
- `parent_idx`（root=0）
- `axis[3]`
- `pivot[3]`
- `motion_range[2]`

#### 3) part_id / ordering / parent-child indexing 统一规则

- token 序中的 part 序号定义为唯一真值（`1..K`）
- `parent_idx` 以该序号编码
- child 可由序号隐式得到（可不单独存 child_idx）
- 所有监督字段都先按同一个 permutation `P` 重排

#### 4) 建议样本结构（用于 joint-aware token supervision）

建议每个样本保存 `npz + json(meta)`：

```text
sample_xxx.npz
  image                uint8    [H,W,3] or float32 [3,518,518]
  mask_raw             int32    [H,W]
  ordered_mask_37      int16    [37,37]
  voxel_coords         int16    [N,3]
  bbox                 float32  [K,2,3]
  joint_type           int8     [K]
  parent_idx           int16    [K]
  axis                 float32  [K,3]
  pivot                float32  [K,3]
  motion_range         float32  [K,2]
  part_perm            int16    [K]      # 原始part -> token序
  input_ids            int32    [L]      # teacher forcing输入
  labels               int32    [L]      # 监督标签（可含ignore_index）

sample_xxx_meta.json
  part_names
  reorder_rule
  quantization_config
  root_policy
```

---

### II.6 你这条路线下最关键的工程风险点

1. `max_length` 当前为 2187，按 16 token/part 很可能不够，需要同步改 `max_length` 与显存预算。
2. `MeshDecodeLogitsProcessor` 当前只约束 6-token bbox 语法，必须扩为 slot-aware grammar。
3. 训练数据里必须固化唯一排序函数；否则 joint graph 会在 token 序上错位。
4. 对 `joint_type` 的条件掩码不做会导致 axis/range 学习噪声过大。

---

### II.7 外挂 joint head 的定位（仅 baseline）

可作为对照实验 baseline：

- `bboxgen -> joint head`

但应明确标注其局限：

- part ordering 对齐在两模块间容易漂移
- 不能天然利用 AR token grammar 约束 joint 输出
- 与你的“统一 token 表示”路线目标不一致

