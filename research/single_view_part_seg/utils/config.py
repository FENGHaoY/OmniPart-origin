from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DatasetConfig:
    data_root: str = "research/outputs/capnet_all_512"
    rgb_dirname: str = "rgb_512"
    seg_dirname: str = "segmentation_512"
    train_split_ratio: float = 0.9
    seed: int = 42


@dataclass
class ModelConfig:
    num_classes: int = 5
    embedding_dim: int = 16
    feat_dim: int = 256
    # "fallback" is lightweight and always available.
    sam_backbone: str = "fallback"   # ["fallback", "sam"]
    dino_backbone: str = "fallback"  # ["fallback", "dinov2", "timm_resnet18"]


@dataclass
class LossConfig:
    ignore_index: int = 255
    alpha_emb: float = 1.0
    use_dice: bool = False
    emb_pull_weight: float = 1.0
    emb_push_weight: float = 1.0
    emb_reg_weight: float = 1e-3
    emb_delta_var: float = 0.5
    emb_delta_dist: float = 1.5


@dataclass
class TrainConfig:
    exp_name: str = "capnet_singapo_baseline"
    output_dir: str = "research/outputs/single_view_part_seg"
    batch_size: int = 4
    num_workers: int = 4
    num_epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-4
    device: str = "cuda"
    save_every: int = 10
    resume_ckpt: Optional[str] = None


@dataclass
class InferConfig:
    ckpt_path: str = ""
    input_rgb: str = ""
    output_dir: str = "research/outputs/single_view_part_seg_infer"
    device: str = "cuda"
    emb_cluster_dist_th: float = 0.35
    min_instance_pixels: int = 30
    max_centers_per_class: int = 4


@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    infer: InferConfig = field(default_factory=InferConfig)

