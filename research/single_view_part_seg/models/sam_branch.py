from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _FallbackSAMEncoder(nn.Module):
    """
    Lightweight substitute for quick debugging when real SAM is unavailable.
    """

    def __init__(self, out_channels: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SAMBranch(nn.Module):
    """
    Wrapper for SAM image encoder.
    - mode="sam": try loading segment-anything image encoder
    - mode="fallback": always use lightweight CNN fallback
    """

    def __init__(
        self,
        mode: str = "fallback",
        out_channels: int = 256,
        sam_ckpt: Optional[str] = None,
        sam_model_type: str = "vit_b",
    ) -> None:
        super().__init__()
        self.mode = mode
        self.encoder: nn.Module
        self.sam_input_size = 1024

        if mode == "sam":
            try:
                from segment_anything import sam_model_registry  # type: ignore

                sam = sam_model_registry[sam_model_type](checkpoint=sam_ckpt)
                self.encoder = sam.image_encoder
                # project SAM channels to target channels
                self.proj = nn.Conv2d(256, out_channels, kernel_size=1)
            except Exception:
                self.encoder = _FallbackSAMEncoder(out_channels=out_channels)
                self.proj = nn.Identity()
                self.mode = "fallback"
        else:
            self.encoder = _FallbackSAMEncoder(out_channels=out_channels)
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Real SAM image encoder expects a fixed token grid (typically from 1024x1024 input).
        # Resize here to avoid positional embedding shape mismatch.
        if self.mode == "sam":
            x = F.interpolate(x, size=(self.sam_input_size, self.sam_input_size), mode="bilinear", align_corners=False)
        feat = self.encoder(x)
        return self.proj(feat)

