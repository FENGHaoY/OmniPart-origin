# OmniPart 两阶段生成：训练/推理数据流与阶段衔接说明

> 仓库：`OmniPart-origin`（你当前 clone 并修改描述文件的版本）
> 重点：阶段一 `bbox_gen` 如何接入、输出、并驱动阶段二

## 1. 总览（先给结论）

OmniPart 在公开代码中的推理链路可以分成：

1. 稀疏结构坐标生成（`get_coords`）
2. 阶段一：`BboxGen` 生成每个 part 的 3D bbox
3. 阶段二：`part_synthesis` 按 bbox 划分点并生成每个 part 的 3D 表示

其中阶段一和阶段二的**关键连接变量**是：

- `voxel_coords.npy`（整体体素坐标）
- `bboxes.npy`（阶段一输出 bbox）
- `part_layouts`（由 `voxel_coords + bboxes` 计算得到，送入阶段二）

---

## 2. 推理数据流（两阶段）

主入口：
- `scripts/inference_omnipart.py`

### 2.1 流程图（简化）

```text
输入图像(RGBA) + 分割mask(.exr, HxWx3 part_id)
        |
        v
load_img_mask()
  -> img_white_bg [3,518,518]
  -> img_black_bg [3,518,518]
  -> ordered_mask_input [37,37]
        |
        v
part_synthesis_pipeline.get_coords(img_black_bg)
  -> voxel_coords [N,4] (batch,x,y,z)
  -> 保存 voxel_coords.npy
        |
        v
prepare_bbox_gen_input(voxel_coords.npy, img_white_bg, ordered_mask_input)
  -> points [1,N,3] (float16, [-0.5,0.5])
  -> whole_voxel_index [1,N,3] (long, 0..63)
  -> images [1,3,518,518]
  -> masks [1,37,37]
        |
        v
阶段一 BboxGen.generate(...)
  -> bboxes [K,2,3] (float, [-0.5,0.5])
  -> 保存 bboxes.npy
        |
        v
prepare_part_synthesis_input(voxel_coords.npy, bboxes.npy, ordered_mask_input)
  -> coords [N_total,4] (int, batch_idx+xyz)
  -> part_layouts [slice0(整体), slice1..sliceK(各part)]
  -> masks [1,37,37]
        |
        v
阶段二 part_synthesis_pipeline.get_slat(...)
  -> mesh / gaussian / radiance_field（分 part 输出）
```

### 2.2 每步对应代码

- 推理主链路：`scripts/inference_omnipart.py`
- 图像与 mask 预处理：`modules/inference_utils.py::load_img_mask`
- 阶段一输入组装：`modules/inference_utils.py::prepare_bbox_gen_input`
- 阶段一模型：`modules/bbox_gen/models/autogressive_bbox_gen.py`
- 阶段一输出解码：`modules/bbox_gen/utils/bbox_tokenizer.py`
- 阶段一->阶段二桥接：`modules/inference_utils.py::prepare_part_synthesis_input`
- 阶段二 pipeline：`modules/part_synthesis/pipelines/omnipart_image_to_parts.py`

---

## 3. 阶段一（BboxGen）细化

### 3.1 阶段一输入

`prepare_bbox_gen_input` 输出字典字段：

- `points`: `[B, M, 3]`，这里推理时 `B=1`
  - 来源 `voxel_coords.npy[:,1:]`
  - 从体素索引 `0..63` 映射到连续坐标 `[-0.5, 0.5]`
- `whole_voxel_index`: `[B, M, 3]`，long，`0..63`
- `images`: `[B, 3, 518, 518]`（白底图）
- `masks`: `[B, 37, 37]`（bottom-up 重排后的 part id）

### 3.2 阶段一模型内部数据形态

在 `BboxGen.generate` 中：

1. 图像编码：DINOv2 -> 图像 token（常见长度 1374）
2. mask embedding：`masks [B,37,37] -> [B,1369,C]`，拼接到图像 token
3. 点云特征：
   - `points` 经 PartField encoder 得到每点 448 维
   - 散射到 `64^3` 体素体 `feat_volume`
   - 再拼上 xyz 三通道 -> `451` 通道输入 3D encoder
   - 下采样后得到 `8^3=512` 个 voxel token
4. 拼接 multimodal token：`voxel_token = image_token + voxel_token`
5. 自回归解码 bbox token 并 decode 为 `[K,2,3]`

### 3.3 阶段一 token 规则（很关键）

- `bins=64`, `BOS=64`, `EOS=65`, `PAD=66`, `vocab_size=67`
- 每个 bbox 用 `2*3=6` 个离散坐标 token
- tokenizer 将连续坐标与离散 token 双向量化：
  - encode: `[-0.5,0.5] -> [0..63]`
  - decode: `[0..63] -> [-0.5,0.5]`

因此阶段一输出 `bboxes.npy` 是连续坐标，格式 `[K,2,3]`。

---

## 4. 两阶段怎么连接

连接逻辑在 `prepare_part_synthesis_input`：

1. 读 `voxel_coords.npy` 得到 `overall_coords [N,3]`（体素索引坐标）
2. 读 `bboxes.npy`，把每个 bbox 从连续空间映射回离散体素区间
3. 用 bbox 区间从 `overall_coords` 挑点，形成每个 part 的点集
4. 记录切片范围 `part_layouts`：
   - 第 0 个 slice：整体
   - 后续 slice：各 part
5. 合并坐标形成 `coords [N_total,4]`（前一列 batch id）
6. 连同 `masks` 一起送入 `get_slat`

阶段二模型在 forward 中显式使用：
- `kwargs['masks']` 做 mask group embedding
- `kwargs['part_layouts']` 做 part 粒度重排与建模

所以：**阶段一不是只给可视化 bbox，它直接决定阶段二的 part 划分与生成范围。**

---

## 5. 训练侧数据流（公开代码）

### 5.1 已公开：阶段二训练（part_synthesis）

训练入口：
- `train.py`
- `configs/training_part_synthesis.json`

README 的训练数据准备（Step1~Step6）最终会得到：
- `all_latent.npz`（整体 + 各 part 的 coords/feats + offsets）
- 渲染条件图和 mask（供 image condition）

数据集 `ImageConditionedSLat` 在 `training/datasets/structured_latent_part.py`：

- 从 `all_latent.npz` 读：
  - `coords`（体素坐标）
  - `feats`（SLat 特征）
  - `offsets`（整体与各 part 分段）
- 构造：
  - `part_layouts`
  - `x_0`（SparseTensor）
  - `cond`（图像条件）
  - `ordered_mask_dino`（`[37,37]`）

训练时 denoiser `structured_latent_flow` 同时吃 `part_layouts + ordered_mask_dino`。

### 5.2 当前仓库未公开：阶段一训练入口

代码里有 `BboxGen.forward`（说明可训练），但我在仓库内没有找到：

- 对应 `bbox_gen` 的 dataset / trainer / config（完整训练入口）
- 给 `input_ids/labels` 的公开训练脚本

这意味着你现在可以直接改阶段一推理与结构，但如果要完整重训阶段一，需要你补训练管线（或从作者后续代码获取）。

---

## 6. 关键数据格式速查

### 6.1 2D 条件

- 输入 mask（README 约定）：`[H, W, 3]` `.exr`，3 通道都存同一个 part_id
- 模型使用的 mask：
  - 推理：`ordered_mask_input [37,37]` long
  - 训练：`ordered_mask_dino [37,37]` long

### 6.2 3D 稀疏坐标

- `voxel_coords`（get_coords 输出）：`[N,4]`，列是 `(batch, x, y, z)`，xyz 整数 `0..63`
- 阶段一 `points`：`[1,N,3]`，连续坐标 `[-0.5,0.5]`
- 阶段一 `whole_voxel_index`：`[1,N,3]`，离散 `0..63`
- 阶段二 `coords`：`[N_total,4]`，用于 SparseTensor

### 6.3 阶段一输出 bbox

- `bboxes.npy`: `[K,2,3]`
  - `K` = part 数（预测）
  - 每个 bbox 两个角点（min/max）
  - 坐标范围约 `[-0.5,0.5]`

---

## 7. 如果你重点改阶段一，建议先盯这 4 个点

1. `prepare_bbox_gen_input`：输入坐标归一化与 mask 编号策略
2. `BboxGen.generate/forward`：图像 token 与 voxel token 融合方式
3. `MeshDecodeLogitsProcessor + BoundsTokenizerDiag`：bbox token 语法与量化精度
4. `prepare_part_synthesis_input`：bbox 到 part_layout 的分配规则（直接影响阶段二）

---

## 8. 你接下来改动时最容易踩的坑

1. `ordered_mask` 的 part 编号顺序必须和你训练/推理一致（这里默认 bottom-up 重排）
2. `bboxes` 坐标系必须保持 `[-0.5,0.5]` 约定，否则阶段二分配会错位
3. `coords` 的第一列是 batch id，别误删
4. 若你改变 bbox 数量或粒度，`part_layouts` 的 slice 连续性要保证

---

## 9. 参考文件（建议按这个顺序读）

1. `scripts/inference_omnipart.py`
2. `modules/inference_utils.py`
3. `modules/bbox_gen/models/autogressive_bbox_gen.py`
4. `modules/bbox_gen/utils/bbox_tokenizer.py`
5. `modules/part_synthesis/pipelines/omnipart_image_to_parts.py`
6. `modules/part_synthesis/models/structured_latent_flow.py`
7. `training/datasets/structured_latent_part.py`
8. `training/models/structured_latent_flow.py`

