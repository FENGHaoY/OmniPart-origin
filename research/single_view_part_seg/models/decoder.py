import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    """
    Simple shared decoder:
    feature map (low-res) -> upsample back to image resolution
    """

    def __init__(self, in_dim: int, mid_dim: int = 128) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(mid_dim, mid_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, out_size: tuple[int, int]) -> torch.Tensor:
        x = self.block1(x)
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.block2(x)
        x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        return x

