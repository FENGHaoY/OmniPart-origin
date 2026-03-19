from typing import Dict, Optional

import torch
import torch.nn as nn

from research.single_view_part_seg.models.decoder import Decoder
from research.single_view_part_seg.models.dino_branch import DinoBranch
from research.single_view_part_seg.models.fusion_module import FeatureAlign, FusionBlock
from research.single_view_part_seg.models.heads import EmbeddingHead, SemanticHead
from research.single_view_part_seg.models.sam_branch import SAMBranch


class PartSegModel(nn.Module):
    """
    Single-view part segmentation model.
    Input:
      image [B,3,H,W]
    Output:
      {
        "pred_semantic": [B,5,H,W],
        "pred_embedding": [B,D,H,W]
      }
    """

    def __init__(
        self,
        num_classes: int = 5,
        embedding_dim: int = 16,
        feat_dim: int = 256,
        sam_backbone: str = "fallback",
        dino_backbone: str = "fallback",
        sam_ckpt: Optional[str] = None,
        sam_model_type: str = "vit_b",
    ) -> None:
        super().__init__()
        self.sam_branch = SAMBranch(
            mode=sam_backbone,
            out_channels=feat_dim,
            sam_ckpt=sam_ckpt,
            sam_model_type=sam_model_type,
        )
        self.dino_branch = DinoBranch(mode=dino_backbone, out_channels=feat_dim)
        self.align = FeatureAlign(in_sam=feat_dim, in_dino=feat_dim, out_dim=feat_dim)
        self.fusion = FusionBlock(feat_dim=feat_dim)
        self.decoder = Decoder(in_dim=feat_dim, mid_dim=feat_dim // 2)
        self.semantic_head = SemanticHead(in_dim=feat_dim // 2, num_classes=num_classes)
        self.embedding_head = EmbeddingHead(in_dim=feat_dim // 2, emb_dim=embedding_dim)

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        h, w = image.shape[-2:]
        f_sam = self.sam_branch(image)
        f_dino = self.dino_branch(image)
        f_sam, f_dino = self.align(f_sam, f_dino)
        f_fuse = self.fusion(f_sam, f_dino)
        f_dec = self.decoder(f_fuse, out_size=(h, w))
        pred_semantic = self.semantic_head(f_dec)
        pred_embedding = self.embedding_head(f_dec)
        return {
            "pred_semantic": pred_semantic,
            "pred_embedding": pred_embedding,
        }

