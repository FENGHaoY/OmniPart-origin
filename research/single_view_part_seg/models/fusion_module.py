import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureAlign(nn.Module):
    def __init__(self, in_sam: int, in_dino: int, out_dim: int) -> None:
        super().__init__()
        self.sam_proj = nn.Conv2d(in_sam, out_dim, kernel_size=1)
        self.dino_proj = nn.Conv2d(in_dino, out_dim, kernel_size=1)

    def forward(self, f_sam: torch.Tensor, f_dino: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Align spatial resolution to dino branch by default.
        if f_sam.shape[-2:] != f_dino.shape[-2:]:
            f_sam = F.interpolate(f_sam, size=f_dino.shape[-2:], mode="bilinear", align_corners=False)
        return self.sam_proj(f_sam), self.dino_proj(f_dino)


class FusionBlock(nn.Module):
    """
    Residual fusion:
      F_fuse = F_dino + lambda * F_sam
    """

    def __init__(self, feat_dim: int) -> None:
        super().__init__()
        self.lambda_param = nn.Parameter(torch.tensor(1.0))
        self.refine = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, f_sam: torch.Tensor, f_dino: torch.Tensor) -> torch.Tensor:
        fused = f_dino + self.lambda_param * f_sam
        return self.refine(fused)

