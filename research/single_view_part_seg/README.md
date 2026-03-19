# 单视图部件级分割模块（CAPNet -> SINGAPO 对齐标签）

本模块用于在你已经处理好的 CAPNet 数据上训练一个“单视图输入”的部件级分割模型。

## 一、整体实现思路

目标是直接训练到统一标签空间（不是 CAPNet 9 类中间态）：

- 输入：单张 RGB 图像
- 模型内部双分支：
  - SAM 分支（区域/边界先验）
  - DINOv2 分支（高层语义）
- 特征对齐 + 残差融合 + 共享解码器
- 两个输出头：
  - 语义分割头（5 类）
  - embedding 头（用于实例分组）

训练监督：

- `L_sem`：`CrossEntropyLoss(ignore_index=255)`（可选 Dice 接口）
- `L_emb`：discriminative embedding loss（pull + push + reg）
- 总损失：`L = L_sem + alpha * L_emb`

参数更新策略（当前默认）：

- **SAM encoder 冻结**
- **DINO encoder 冻结**
- 仅训练对齐层、融合层、decoder 与两个 head
- 若要放开，可在训练命令中追加 `--unfreeze_sam` 或 `--unfreeze_dino`

推理：

- `pred_semantic` 取 `argmax` 得到语义图
- 结合 `pred_embedding` 做类内实例分组，输出实例图和实例语义

## 二、输入与输出

### 1) 数据输入（训练）

默认读取目录（示例）：

- `research/outputs/capnet_all_512/rgb_512/*.png`
- `research/outputs/capnet_all_512/segmentation_512/*.npz`

每个 `.npz` 至少包含：

- `semantic_segmentation`
- `instance_segmentation`

### 2) Dataset 样本输出

`dataset/capnet_singapo_dataset.py` 每条样本输出：

- `image`: `Tensor[3,H,W]`
- `gt_semantic`: `Tensor[H,W]`（已映射到 5 类空间）
- `gt_instance`: `Tensor[H,W]`（ignore 区域置 255）
- `valid_mask`: `Tensor[H,W]`（`gt_semantic != 255`）
- `meta`: 图像路径、分割路径、原尺寸等

### 3) 模型输出

- `pred_semantic`: `[B,5,H,W]`
  - `0 base`, `1 door`, `2 drawer`, `3 handle`, `4 knob`
- `pred_embedding`: `[B,D,H,W]`（默认 `D=16`）

## 三、标签空间与映射

统一训练标签：

- `0 base`
- `1 door`
- `2 drawer`
- `3 handle`
- `4 knob`
- `255 ignore`

CAPNet -> 统一标签映射已在 `utils/label_mapping.py` 固化实现，训练时直接使用该映射监督，不走 CAPNet 9 类训练再映射。

## 四、简化假设与可替换接口

已明确保留可替换接口，当前 baseline 优先“可跑 + 清晰”：

- `models/sam_branch.py`
  - `mode=fallback`：轻量 CNN，快速调试
  - `mode=sam`：真实 SAM image encoder（需要权重）
- `models/dino_branch.py`
  - `mode=fallback`：轻量 CNN
  - `mode=dinov2`：尝试用 `torch.hub` 加载 DINOv2（会优先走本地缓存）
  - `mode=timm_resnet18`：轻量语义 backbone 备选

> 说明：真实 SAM / DINOv2 受环境和权重可用性影响；因此默认使用 fallback 保证流程稳定跑通。

## 五、如何替换成真实 SAM / DINOv2

### 1) 真实 SAM

你的 SAM 权重路径：

- `/home/bingxing2/home/scx8q10/xiaoqian/OmniPart/ckpt/sam_vit_h_4b8939.pth`

SAM 模型类型可通过参数指定：`--sam_model_type vit_b|vit_l|vit_h`。
你的权重是 `vit_h`，建议显式传 `--sam_model_type vit_h`。

训练命令（真实 SAM，模块方式推荐）：

```bash
conda activate omnipart
python -m research.single_view_part_seg.train \
  --data_root research/outputs/capnet_all_512 \
  --output_dir research/outputs/single_view_part_seg_real \
  --epochs 20 \
  --batch_size 4 \
  --lr 1e-4 \
  --alpha 1.0 \
  --sam_backbone sam \
  --sam_model_type vit_h \
  --sam_ckpt /home/bingxing2/home/scx8q10/xiaoqian/OmniPart/ckpt/sam_vit_h_4b8939.pth \
  --dino_backbone dinov2
```

若你坚持脚本方式，请用（避免 `No module named research`）：

```bash
cd /home/bingxing2/home/scx8q10/xiaoqian/OmniPart-origin
PYTHONPATH=/home/bingxing2/home/scx8q10/xiaoqian/OmniPart-origin \
python research/single_view_part_seg/train.py \
  --data_root research/outputs/capnet_all_512 \
  --output_dir research/outputs/single_view_part_seg_real \
  --epochs 20 \
  --batch_size 4 \
  --lr 1e-4 \
  --alpha 1.0 \
  --sam_backbone sam \
  --sam_model_type vit_h \
  --sam_ckpt /home/bingxing2/home/scx8q10/xiaoqian/OmniPart/ckpt/sam_vit_h_4b8939.pth \
  --dino_backbone dinov2
```

推理时也要传 `--sam_ckpt`（`infer.py` 已支持）：

```bash
conda activate omnipart
python research/single_view_part_seg/infer.py \
  --ckpt research/outputs/single_view_part_seg_real/best.pt \
  --sam_model_type vit_h \
  --sam_ckpt /home/bingxing2/home/scx8q10/xiaoqian/OmniPart/ckpt/sam_vit_h_4b8939.pth \
  --rgb research/outputs/capnet_all_512/rgb_512/StorageFurniture_46179_0_0.png \
  --npz research/outputs/capnet_all_512/segmentation_512/StorageFurniture_46179_0_0.npz \
  --out_dir research/outputs/single_view_part_seg_infer_real
```

### 2) 真实 DINOv2

- `dino_branch.py` 的 `mode=dinov2` 使用 `torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")`
- 若你机器上已有缓存，会直接读取缓存；否则会尝试在线下载

你提供的 DINOv2 本地缓存路径为：

- `~/.cache/torch/hub/checkpoints/dinov2_vitb14_reg4_pretrain.pth`

- `--dino_backbone dinov2`

## 六、快速运行指南

### 1) 快速 baseline（推荐先跑通）

```bash
conda activate omnipart
python -m research.single_view_part_seg.train \
  --data_root research/outputs/capnet_all_512 \
  --output_dir research/outputs/single_view_part_seg \
  --epochs 20 \
  --batch_size 4 \
  --save_every 1 \
  --lr 1e-4 \
  --alpha 1.0 \
  --sam_backbone fallback \
  --dino_backbone fallback
```

### 2) 真实 SAM + 真实 DINOv2（走本地路径）

> 说明：当前代码中 DINOv2 通过 `torch.hub` 加载，读取的是 `torch hub` 缓存目录。  
> 你的 DINOv2 权重已在 `~/.cache/torch/hub/checkpoints/dinov2_vitb14_reg4_pretrain.pth`，因此只需设置 `--dino_backbone dinov2` 即可触发加载。

```bash
conda activate omnipart
python -m research.single_view_part_seg.train \
  --data_root research/outputs/capnet_all_512 \
  --output_dir research/outputs/single_view_part_seg_real \
  --epochs 20 \
  --batch_size 4 \
  --save_every 10 \
  --lr 1e-4 \
  --alpha 1.0 \
  --sam_backbone sam \
  --sam_model_type vit_h \
  --sam_ckpt /home/bingxing2/home/scx8q10/xiaoqian/OmniPart/ckpt/sam_vit_h_4b8939.pth \
  --dino_backbone dinov2
```

### 3) 单图推理（模块方式，推荐，真实 SAM vit-h + DINOv2）

```bash
conda activate omnipart
python -m research.single_view_part_seg.infer \
  --ckpt research/outputs/single_view_part_seg_real/best.pt \
  --sam_model_type vit_h \
  --sam_ckpt /home/bingxing2/home/scx8q10/xiaoqian/OmniPart/ckpt/sam_vit_h_4b8939.pth \
  --rgb research/outputs/capnet_all_512/rgb_512/StorageFurniture_46179_0_0.png \
  --npz research/outputs/capnet_all_512/segmentation_512/StorageFurniture_46179_0_0.npz \
  --out_dir research/outputs/single_view_part_seg_infer
```

> 若 checkpoint 里已包含完整 SAM 分支权重，`--sam_ckpt` 通常可省略；`--sam_model_type vit_h` 建议保留以避免配置不一致。

### 4) 单图推理（脚本方式，真实 SAM vit-h + DINOv2）

> 需要先在仓库根目录执行（否则可能出现 `No module named 'research'`）：
>
> `cd /home/bingxing2/home/scx8q10/xiaoqian/OmniPart-origin`

```bash
conda activate omnipart
PYTHONPATH=/home/bingxing2/home/scx8q10/xiaoqian/OmniPart-origin \
python research/single_view_part_seg/infer.py \
  --ckpt research/outputs/single_view_part_seg_real/best.pt \
  --sam_model_type vit_h \
  --sam_ckpt /home/bingxing2/home/scx8q10/xiaoqian/OmniPart/ckpt/sam_vit_h_4b8939.pth \
  --rgb research/outputs/capnet_all_512/rgb_512/StorageFurniture_46179_0_0.png \
  --npz research/outputs/capnet_all_512/segmentation_512/StorageFurniture_46179_0_0.npz \
  --out_dir research/outputs/single_view_part_seg_infer
```

推理可视化会保存：

- 输入图
- GT/PRED semantic
- GT/PRED instance

## 七、常见报错排查

### 1) SAM 权重与模型类型不匹配

现象：加载 SAM 报错（shape/key mismatch）。  
排查：

- 你当前权重是 `sam_vit_h_4b8939.pth`，必须配 `--sam_model_type vit_h`
- 若改用其它权重，同步修改 `--sam_model_type`

### 2) DINOv2 下载/加载失败

现象：`torch.hub.load(...)` 报网络错误或仓库拉取失败。  
排查：

- 先确认本地存在：`~/.cache/torch/hub/checkpoints/dinov2_vitb14_reg4_pretrain.pth`
- 网络受限时可先在可联网环境完成一次 `torch.hub` 缓存
- 兜底方案：先用 `--dino_backbone fallback` 跑通训练流程

### 3) CUDA 显存不足（OOM）

排查建议：

- 降低 `--batch_size`（例如 4 -> 2 -> 1）
- 先用 `--sam_backbone fallback --dino_backbone fallback`
- 必要时减小输入分辨率（如果你后续加了 resize 策略）

### 4) 数据读取报错（找不到 rgb/segmentation）

排查：

- 确认目录下是 `rgb_512/` 与 `segmentation_512/`
- `segmentation_512/*.npz` 与 `rgb_512/*.png` stem 必须一一对应
- `.npz` 至少有 `semantic_segmentation` 和 `instance_segmentation`

### 5) `-m` 与脚本路径混用

错误写法（不要这样）：

- `python -m research/single_view_part_seg/infer.py`

正确写法二选一：

- 模块方式：`python -m research.single_view_part_seg.infer ...`
- 脚本方式：`python research/single_view_part_seg/infer.py ...`

### 6) checkpoint 保存失败（`PytorchStreamWriter failed writing file`）

现象：训练能跑到某个 epoch，但在 `torch.save` 时崩溃。  
常见原因：磁盘配额不足、NFS 写入失败、临时 I/O 抖动。

已做的保护：

- 训练脚本改为“先写 `.tmp` 再 `os.replace`”的稳健保存
- 保存失败时会自动尝试写更小的 `*_model_only.pt`

建议：

- 减少保存频率：例如 `--save_every 10`
- 检查输出目录可用空间/配额