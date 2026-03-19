from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _FallbackDINOEncoder(nn.Module):
    def __init__(self, out_channels: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DinoBranch(nn.Module):
    """
    DINOv2 branch wrapper.
    - mode="dinov2": try torch.hub facebookresearch/dinov2
    - mode="timm_resnet18": use timm backbone features as semantic branch substitute
    - mode="fallback": lightweight CNN
    """

    def __init__(self, mode: str = "fallback", out_channels: int = 256) -> None:
        super().__init__()
        self.mode = mode
        self.encoder: nn.Module
        self.proj: nn.Module

        if mode == "dinov2":
            try:
                # Returns token features; this branch intentionally keeps a fallback path
                self.encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
                self.proj = nn.Conv2d(384, out_channels, kernel_size=1)
            except Exception:
                self.encoder = _FallbackDINOEncoder(out_channels=out_channels)
                self.proj = nn.Identity()
                self.mode = "fallback"
        elif mode == "timm_resnet18":
            try:
                import timm  # type: ignore

                self.encoder = timm.create_model("resnet18", pretrained=True, features_only=True)
                # resnet18 last feature has 512 channels
                self.proj = nn.Conv2d(512, out_channels, kernel_size=1)
            except Exception:
                self.encoder = _FallbackDINOEncoder(out_channels=out_channels)
                self.proj = nn.Identity()
                self.mode = "fallback"
        else:
            self.encoder = _FallbackDINOEncoder(out_channels=out_channels)
            self.proj = nn.Identity()

    def _forward_dinov2(self, x: torch.Tensor) -> torch.Tensor:
        # DINOv2 ViT models are typically trained/evaluated around 518 resolution.
        # Resize to a stable size so patch-token grid is well-defined.
        x = F.interpolate(x, size=(518, 518), mode="bilinear", align_corners=False)
        # DINOv2 forward returns tokens [B, N, C] for ViT; convert to spatial map.
        tokens = self.encoder.forward_features(x)
        if isinstance(tokens, dict) and "x_norm_patchtokens" in tokens:
            t = tokens["x_norm_patchtokens"]  # [B, N, C]
        else:
            t = tokens  # best effort
        b, n, c = t.shape
        h = w = int(n ** 0.5)
        feat = t[:, : h * w, :].transpose(1, 2).reshape(b, c, h, w)
        return self.proj(feat)

    def _forward_timm(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        feat = feats[-1]
        return self.proj(feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "dinov2":
            return self._forward_dinov2(x)
        if self.mode == "timm_resnet18":
            return self._forward_timm(x)
        return self.proj(self.encoder(x))

