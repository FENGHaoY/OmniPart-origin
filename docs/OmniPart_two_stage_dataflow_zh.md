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

### 2.3 用你训练的“部件级分割模块”替换 SAM（接入点与数据约定）

公开版 OmniPart 的推理链路**本质上只依赖一个“2D part-id mask”**（再被下采样/重排为 `ordered_mask_input [37,37]`），并不强绑定 SAM。原工程很多复现流程里用 SAM 产生该 mask；你现在训练好了自己的部件级分割模块，可以按以下方式替换：

#### A. 你需要产出的最小中间结果是什么？

- **最小必需**：一个与输入图像对齐的 **part_id map**（每像素一个整数 id）
  - 语义：`0=背景`，`1..K=部件编号`
  - 推理侧最终会被 `modules/inference_utils.py::load_img_mask` 转成 `ordered_mask_input [37,37]`（long）
- **与当前文档保持一致的落盘格式**（推荐沿用，改动最少）：
  - `.exr`：形状 `[H,W,3]`，三个通道都写同一个 `part_id`（原 README 约定）
  - 或者你也可以在 `load_img_mask` 支持读取单通道 png（如果你后续愿意改代码；本条仅说明）

#### B. 应该加到哪里（推理时机/位置）

你要替换的“SAM 生成 mask”发生在两阶段推理之前，属于**输入预处理**。因此你的部件级分割模块应当接在：

- `scripts/inference_omnipart.py` 中调用 `load_img_mask()` 之前

具体建议两种接法（只写思路，不在此处实现代码）：

1. **离线/外部预处理（推荐，最简单稳妥）**
   - 你先用自己的分割网络对输入图像跑一遍，输出 `part_id.exr`（或等价格式）
   - 然后保持 `scripts/inference_omnipart.py` 不变：直接把“图像 + mask 路径”喂进去
   - 优点：最少侵入主推理代码；方便缓存与复现；训练与推理共用同一套 mask 文件

2. **在线推理内调用（更像 end-to-end demo）**
   - 在 `scripts/inference_omnipart.py` 里新增一个分支：若未提供 mask 路径，则调用你的分割模块生成 `part_id map`，再交给 `load_img_mask`（或直接生成 `ordered_mask_input`）
   - 优点：用户只需给一张图
   - 风险点：要确保分割模块的输入尺度/裁剪/alpha 处理与 `load_img_mask` 的图像预处理保持一致，否则 mask 与图像对不齐

#### C. 关键兼容点（必须对齐，否则阶段一/二都会“看错”）

1. **编号语义与数量**
   - OmniPart 后续只看一个整数 mask（最后变成 `[37,37]`），因此你必须在推理时确定 `K`（部件数）
   - 你的网络若输出类别通道：建议直接 `argmax` 得到 `part_id map`

2. **`ordered_mask_input` 的“bottom-up 重排”**
   - 文档中的 `ordered_mask_input` 是经过 `load_img_mask` 的重排/下采样得到的 `37×37` mask
   - 因此你训练分割网络时的 **part_id 顺序**要与推理时一致；否则即使分割对了，阶段一/二拿到的“部件编号语义”也会错位
   - 若你自己的网络输出的是“无序实例”（类似 instance id），则需要在输出 `.exr` 前定义稳定的排序策略（例如按面积、按 bbox x 坐标、按 tree/part_order 等）

##### C.2.1 无序实例 id → 稳定 part_id 的几套高效方案（按推荐顺序）

你的问题本质是：**分割模型输出的是“实例集合”**（无序、数量可变），而 OmniPart 下游希望一个**有序且语义稳定**的 `part_id`（最好能对齐到 Singapo 的 `object_meta.parts[]`：`part_id/parent_id/name`）。

下面给出几套方案，从“几乎不改训练”到“强对齐 object_meta tree”，你可以按自己的资源/目标选择。

**方案 1（最低成本，强稳定）：按 object_meta 的 `name→part_id` 做语义分类式输出**

- **前提**：你能把部件级分割训练成 *semantic part segmentation*（类别集合 = `object_meta.parts[].name` 的规范化词表，比如 dishwasher: {base, door, handle}）。
- **做法**：
  - 模型直接输出 `K` 类（+background），推理取 `argmax` 得到 `part_id map`（按固定 label index 编号）
  - label index 的定义直接取 `object_meta.part_order`（或你制定的固定顺序）
- **优点**：不需要“重排”；输出天然有序；与训练/推理最一致。
- **缺点**：需要有语义标签/词表；跨类/跨形态需要做 label 归一（同义词/别名）。

**方案 2（有语义但模型仍是实例）：实例 → 语义标签 → part_id 映射**

- **前提**：你的分割模块能为每个实例给出语义标签（显式分类头、或额外的 region classifier）。
- **做法**：
  1. 对每个实例 \(i\) 得到一个语义 label（例如 door/handle/base）。
  2. 通过一个**固定字典**把 label 映射到 `object_meta` 的 part（例如 `handle→part_id=2`）。
  3. 若同一 label 出现多个实例（比如多个 drawer），用子规则排序并编号（见方案 4）。
- **优点**：保留实例分割能力，同时能对齐到 `object_meta`。
- **缺点**：需要“label→part”规范化；同类多实例时仍需二级排序。

**方案 3（无需显式语义标签，工程上很快）：用 region 特征做“文本/原型匹配”得到 name**

- **前提**：你能提 region feature（例如你在 pseudo seg pipeline 里已实现的 DINOv2 pooled feature），并能准备每个 `name` 的“原型向量”。  
  原型向量可以来自：
  - 少量人工标注样本：每个 `name` 选一些 region feature 求均值当原型；
  - 或用图文模型（如 CLIP）做 `text(name)` 与 region/image crop 相似度（工程上更重，但不需要标注太多）。
- **做法**：
  1. 对每个实例算 feature \(f_i\)，对每个 `object_meta.parts[].name` 有原型 \(p_j\)。
  2. 用余弦相似度 \(s_{ij}\) 做匹配，得到实例的 `name`，再映射到 `part_id`。
  3. 可加几何先验修正：base 通常最大、handle 细长、door 贴前表面等。
- **优点**：少标注、可快速落地；适合“实例无序但可语义对齐”的场景。
- **缺点**：语义相近的部件可能混淆；需要每类/每 name 的原型或文本提示工程。

**方案 4（不依赖语义，适合多实例）：纯几何/2D 排序规则生成稳定顺序**

- **前提**：下游不要求严格对齐 `name`，只要顺序稳定（训练/推理一致），即可让模型学到一致的“编号语义”。  
  这是很多“无监督/弱监督”流程最常用的兜底。
- **做法（常用且稳定的排序键）**：
  - **一级**：实例面积从大到小（先把 base/主体排前）
  - **二级**：实例 2D bbox 中心 \(x\)（左→右），再 \(y\)（上→下）
  - **三级**：实例 bbox 的宽高比（细长件如 handle 可稳定靠后/靠前）
- **输出**：按该排序把实例 id 重新编号为 1..K（K 可截断为 `object_meta` 的 part 数，或保留前 K 个最大实例）。
- **优点**：最简单、无需额外模型；跨数据稳定。
- **缺点**：不能保证与 `object_meta.part_id/name` 对齐；更像“稳定编号的 instance mask”。

**方案 5（强对齐 Singapo 结构）：用 `parent_id` 构建 tree，并做“树一致”的匹配/排序**

- **前提**：你希望最终编号与 `object_meta` 一致（尤其是 parent-child 关系要对上，例如 base→door→handle），并且可以使用 object_meta 的结构先验。
- **做法（建议用匈牙利匹配/动态规划的思想，成本不高）**：
  1. 从 `object_meta` 取出目标树：节点集合 \(P=\{p_j\}\)（含 `name`, `parent_id`, 是否 articulated 等）。
  2. 从分割实例得到候选集合 \(R=\{r_i\}\)（每个有 mask、2D bbox、area、feature）。
  3. 定义一个匹配代价 \(C_{ij}\)，综合：
     - **语义代价**（若有）：`name` 一致性 / 文本-特征相似度
     - **几何代价**：与父节点的包含关系（child 应主要落在 parent 的 2D bbox 附近/内部）、相对位置（handle 在 door 上）、尺寸比例（handle << door）
     - **可动性先验**：articulated（如 door）往往是大块且与 base 接壤
  4. 在每层（或整体）做最小代价匹配，得到 `实例 i → part j`。
  5. 最终编号直接输出 `part_order` 的顺序（保证与 Singapo 一致）。
- **优点**：对齐最强，且可解释；能把“无序实例”变成“有序且结构一致”的 part_id。
- **缺点**：需要你实现一套代价与匹配（但工程量通常不大）；极端遮挡视角下可能不稳定，需要 fallback。

方案 1A（最推荐、最稳）：把 door 语义拆成 door_0/door_1/...（训练时就固定编号）
思路：不要只学 door，而是学 door_left/door_right 或 door_0/door_1 这种带序号的语义类。
怎么编号：用 object-level 的稳定规则定义 door_0/door_1，常用的是：
按 3D tree / part_order（如果 object_meta 已经给了两个不同的 part_id 且各自 parent/children 不同）
或 按 2D 位置（bbox center x 左→右，或上→下）作为“门 0/门 1”
优点：推理阶段不需要再做匹配；最稳定。
缺点：训练标签要提供“哪个门是 0/1”的监督（或你要自己在数据准备阶段生成这个编号）。
方案 1B（不改训练标签，推理时对齐）：语义分割 + 连通域实例化 + 组内排序
步骤：
语义分割得到 door 的像素集合。
对 door 类做 connected components（或简单分水岭），得到 door 的多个实例区域。
用稳定排序键给这些实例编号：例如 bbox center x（左→右）、再 y（上→下）、再面积（大→小）。
将排序后的 door#0, door#1 对应到 object_meta 里两个 door 的 part_id（需要你定义 object_meta 的 door 子序，如也按 x 排）。
优点：不需要实例分割模型；实现快。
缺点：如果两个门紧挨/遮挡严重，连通域可能粘连；此时要加分割后处理或 fallback。
方案 1C（最强对齐 object_meta）：同语义组内做“匹配”，而不是排序
适用：你想让 “door A ↔ object_meta 的 door part_id=5” 这种对应更可信（尤其有 parent/children 结构差异时）。
做法：
对同语义的多个实例 (r_i) 与 object_meta 里同语义的多个 part (p_j) 构造代价矩阵 (C_{ij})，代价可用：
相对 parent 的位置关系（两个 door 各自更接近哪个 parent 区域）
与 handle/knob 等 child 的共现（哪个 door 上有 handle 的区域重叠/更近）
2D 几何先验（左门/右门、上门/下门）
用匈牙利匹配得到一一对应，然后按 part_order 输出最终 part_id。
优点：当 object_meta 的树结构信息足够时，对齐最强。
缺点：实现略复杂，但通常只在“同语义多实例”这一小块上做匹配，成本不高。




**推荐落地顺序（经验）**：

1. 若你能做成语义分割：优先 **方案 1**（最干净）。
2. 若你现在是实例分割但能出语义：用 **方案 2 + 4（多实例二级排序）**。
3. 若你没有语义但想快速跑通：先用 **方案 4** 保证“编号稳定”，再逐步加语义/结构。
4. 若你明确要对齐 `object_meta.part_id/parent_id/name`：直接做 **方案 5**（可先用粗糙代价 + fallback，逐步精炼）。

3. **分辨率与对齐**
   - `load_img_mask` 会把图像处理到 `518×518` 并同步处理 mask
   - 若你走离线 `.exr`：务必保证该 mask 与原图像像素对齐（同 H/W、同裁剪）

#### D. 训练数据准备侧要怎么替换

文档第 5 节提到阶段二训练数据需要“渲染条件图和 mask（供 image condition）”。如果原流程用 SAM 生成 mask，那么你现在应当：

- 用你的分割模块批量产出与训练图像对应的 `part_id.exr`（或等价 mask）
- 训练 dataset（如 `training/datasets/structured_latent_part.py`）最终仍需要 `ordered_mask_dino [37,37]`：
  - 要么沿用现有读取/下采样逻辑（推荐：与推理一致）
  - 要么你在数据准备阶段就直接存 `37×37` 的 ordered mask（但要确保完全复现 `load_img_mask` 的规则）

一句话总结：**你替换的是“mask 来源”，不是两阶段模型内部接口**；只要你能稳定产出与图像对齐、编号一致的 `part_id` mask，并能被 `load_img_mask` 转成 `ordered_mask_input [37,37]`，就能无缝替换 SAM。

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

> 备注：原流程可用 SAM 生成该 `.exr`，你现在可用自训练的部件级分割模块生成同格式的 `.exr` 来替换，见 2.3 节。

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



1. `scripts/inference_omnipart.py`
2. `modules/inference_utils.py`
3. `modules/bbox_gen/models/autogressive_bbox_gen.py`
4. `modules/bbox_gen/utils/bbox_tokenizer.py`
5. `modules/part_synthesis/pipelines/omnipart_image_to_parts.py`
6. `modules/part_synthesis/models/structured_latent_flow.py`
7. `training/datasets/structured_latent_part.py`
8. `training/models/structured_latent_flow.py`

