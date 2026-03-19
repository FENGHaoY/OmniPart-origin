from typing import Dict

import torch


@torch.no_grad()
def pixel_accuracy(pred: torch.Tensor, target: torch.Tensor, ignore_index: int = 255) -> float:
    """
    pred: [B,H,W] int
    target: [B,H,W] int
    """
    valid = target != ignore_index
    denom = valid.sum().item()
    if denom == 0:
        return 0.0
    correct = ((pred == target) & valid).sum().item()
    return float(correct / denom)


@torch.no_grad()
def mean_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 5,
    ignore_index: int = 255,
) -> float:
    ious = []
    valid = target != ignore_index
    for c in range(num_classes):
        p = (pred == c) & valid
        t = (target == c) & valid
        inter = (p & t).sum().item()
        union = (p | t).sum().item()
        if union == 0:
            continue
        ious.append(inter / union)
    if not ious:
        return 0.0
    return float(sum(ious) / len(ious))


def metric_dict(pred: torch.Tensor, target: torch.Tensor, ignore_index: int = 255) -> Dict[str, float]:
    return {
        "pixel_acc": pixel_accuracy(pred, target, ignore_index=ignore_index),
        "miou": mean_iou(pred, target, num_classes=5, ignore_index=ignore_index),
    }

