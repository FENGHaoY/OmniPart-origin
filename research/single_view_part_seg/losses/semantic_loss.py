import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticLoss(nn.Module):
    def __init__(self, ignore_index: int = 255, use_dice: bool = False, dice_weight: float = 0.3) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.ignore_index = ignore_index
        self.use_dice = use_dice
        self.dice_weight = dice_weight

    def dice_loss(self, logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        valid = target != self.ignore_index
        loss = logits.new_tensor(0.0)
        eps = 1e-6

        for c in range(num_classes):
            p = probs[:, c]
            t = (target == c).float()
            p = p * valid.float()
            t = t * valid.float()
            inter = (p * t).sum(dim=(1, 2))
            denom = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
            dice = (2.0 * inter + eps) / (denom + eps)
            loss = loss + (1.0 - dice.mean())
        return loss / float(num_classes)

    def forward(self, pred_semantic: torch.Tensor, gt_semantic: torch.Tensor) -> torch.Tensor:
        ce = self.ce(pred_semantic, gt_semantic.long())
        if not self.use_dice:
            return ce
        num_classes = pred_semantic.shape[1]
        dice = self.dice_loss(pred_semantic, gt_semantic, num_classes=num_classes)
        return ce + self.dice_weight * dice

