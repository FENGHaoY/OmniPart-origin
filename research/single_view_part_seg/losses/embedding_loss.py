from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscriminativeEmbeddingLoss(nn.Module):
    """
    Pixel-level discriminative embedding loss:
      - pull: pixels should be close to their instance center
      - push: different instance centers should be far apart
      - reg: keep centers near origin
    Only valid pixels are used (valid_mask=True and instance != ignore_index).
    """

    def __init__(
        self,
        delta_var: float = 0.5,
        delta_dist: float = 1.5,
        pull_weight: float = 1.0,
        push_weight: float = 1.0,
        reg_weight: float = 1e-3,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.pull_weight = pull_weight
        self.push_weight = push_weight
        self.reg_weight = reg_weight
        self.ignore_index = ignore_index

    @staticmethod
    def _instance_centers(emb: torch.Tensor, inst_ids: torch.Tensor) -> tuple[List[torch.Tensor], List[int]]:
        """
        emb: [N, D], inst_ids: [N]
        """
        unique_ids = torch.unique(inst_ids)
        centers: List[torch.Tensor] = []
        valid_ids: List[int] = []
        for iid in unique_ids:
            mask = inst_ids == iid
            if mask.sum() == 0:
                continue
            centers.append(emb[mask].mean(dim=0))
            valid_ids.append(int(iid.item()))
        return centers, valid_ids

    def forward(
        self,
        pred_embedding: torch.Tensor,  # [B,D,H,W]
        gt_instance: torch.Tensor,     # [B,H,W]
        valid_mask: torch.Tensor,      # [B,H,W] bool
    ) -> torch.Tensor:
        b, d, h, w = pred_embedding.shape
        total_pull = pred_embedding.new_tensor(0.0)
        total_push = pred_embedding.new_tensor(0.0)
        total_reg = pred_embedding.new_tensor(0.0)
        valid_batches = 0

        emb = pred_embedding.permute(0, 2, 3, 1).contiguous()  # [B,H,W,D]
        for bi in range(b):
            m = valid_mask[bi] & (gt_instance[bi] != self.ignore_index)
            if m.sum() == 0:
                continue

            e = emb[bi][m]            # [N,D]
            inst = gt_instance[bi][m] # [N]
            centers, ids = self._instance_centers(e, inst)
            if len(centers) == 0:
                continue

            centers_t = torch.stack(centers, dim=0)  # [K,D]

            # Pull loss
            pull = e.new_tensor(0.0)
            for k, iid in enumerate(ids):
                pix = e[inst == iid]
                if pix.shape[0] == 0:
                    continue
                dist = torch.norm(pix - centers_t[k : k + 1], dim=1)
                pull = pull + torch.mean(F.relu(dist - self.delta_var) ** 2)
            pull = pull / max(len(ids), 1)

            # Push loss
            if len(ids) > 1:
                c1 = centers_t.unsqueeze(0)  # [1,K,D]
                c2 = centers_t.unsqueeze(1)  # [K,1,D]
                dist_mat = torch.norm(c1 - c2, dim=-1)  # [K,K]
                eye = torch.eye(len(ids), device=dist_mat.device, dtype=torch.bool)
                push_mat = F.relu(2 * self.delta_dist - dist_mat) ** 2
                push = push_mat[~eye].mean()
            else:
                push = e.new_tensor(0.0)

            reg = torch.norm(centers_t, dim=1).mean()

            total_pull = total_pull + pull
            total_push = total_push + push
            total_reg = total_reg + reg
            valid_batches += 1

        if valid_batches == 0:
            return pred_embedding.new_tensor(0.0)

        total_pull = total_pull / valid_batches
        total_push = total_push / valid_batches
        total_reg = total_reg / valid_batches
        return (
            self.pull_weight * total_pull
            + self.push_weight * total_push
            + self.reg_weight * total_reg
        )

