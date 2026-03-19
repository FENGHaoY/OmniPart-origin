import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

from research.single_view_part_seg.dataset.capnet_singapo_dataset import CapnetSingapoDataset
from research.single_view_part_seg.models.part_seg_model import PartSegModel
from research.single_view_part_seg.utils.label_mapping import (
    IGNORE_INDEX,
    filter_instance_with_valid_mask,
    map_capnet_semantic_to_unified,
)
from research.single_view_part_seg.utils.visualization import save_infer_visuals

try:
    from scipy import ndimage as ndi  # type: ignore
except Exception:
    ndi = None


def _normalize_emb(emb: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    n = np.linalg.norm(emb, axis=-1, keepdims=True)
    return emb / np.maximum(n, eps)


def _classwise_instance_grouping(
    pred_sem: np.ndarray,        # [H,W]
    pred_emb: np.ndarray,        # [H,W,D]
    emb_cluster_dist_th: float,
    min_instance_pixels: int,
    max_centers_per_class: int,
) -> Tuple[np.ndarray, List[Dict[str, int]]]:
    """
    Simple instance grouping:
      1) group by semantic class
      2) connected-components pre-split
      3) optional embedding-based split inside each component
    """
    h, w = pred_sem.shape
    instance_map = np.full((h, w), 255, dtype=np.int64)
    instances: List[Dict[str, int]] = []
    next_id = 0

    if ndi is None:
        # Fallback: no scipy -> only semantic connected-components unavailable, use one instance per class mask
        for c in range(5):
            m = pred_sem == c
            if int(m.sum()) < min_instance_pixels:
                continue
            instance_map[m] = next_id
            instances.append({"instance_id": next_id, "semantic_class": int(c), "area": int(m.sum())})
            next_id += 1
        return instance_map, instances

    emb_norm = _normalize_emb(pred_emb)
    struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)

    for c in range(5):
        mask_c = pred_sem == c
        if not mask_c.any():
            continue
        cc_map, cc_num = ndi.label(mask_c, structure=struct)
        for cc_id in range(1, cc_num + 1):
            cc_mask = cc_map == cc_id
            if int(cc_mask.sum()) < min_instance_pixels:
                continue
            pix = emb_norm[cc_mask]  # [N,D]

            # lightweight center discovery by distance threshold
            centers = [pix[0]]
            for i in range(1, pix.shape[0], max(1, pix.shape[0] // 2000)):
                d = np.linalg.norm(pix[i][None, :] - np.stack(centers, axis=0), axis=1).min()
                if d > emb_cluster_dist_th and len(centers) < max_centers_per_class:
                    centers.append(pix[i])

            centers_np = np.stack(centers, axis=0)  # [K,D]
            dmat = np.linalg.norm(pix[:, None, :] - centers_np[None, :, :], axis=-1)
            assign = np.argmin(dmat, axis=1)

            ys, xs = np.where(cc_mask)
            for k in range(centers_np.shape[0]):
                sub = assign == k
                if int(sub.sum()) < min_instance_pixels:
                    continue
                yy = ys[sub]
                xx = xs[sub]
                instance_map[yy, xx] = next_id
                instances.append(
                    {"instance_id": next_id, "semantic_class": int(c), "area": int(sub.sum())}
                )
                next_id += 1

    return instance_map, instances


def _load_rgb(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--rgb", default=None, help="Single image inference")
    parser.add_argument("--npz", default=None, help="Optional GT npz for visualization")
    parser.add_argument("--data_root", default=None, help="Run over validation split if set")
    parser.add_argument("--out_dir", default="research/outputs/single_view_part_seg_infer")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sam_ckpt", default=None, help="SAM checkpoint path when model uses real SAM branch.")
    parser.add_argument("--sam_model_type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--emb_dist_th", type=float, default=0.35)
    parser.add_argument("--min_pixels", type=int, default=30)
    parser.add_argument("--max_centers_per_class", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt.get("config", {})
    mcfg = cfg.get("model", {})
    ckpt_model = ckpt.get("model", {})

    # Auto-infer SAM model type from checkpoint to avoid vit_b/vit_l/vit_h mismatch.
    sam_model_type = args.sam_model_type
    if mcfg.get("sam_backbone", "fallback") == "sam":
        w = ckpt_model.get("sam_branch.encoder.patch_embed.proj.weight", None)
        if isinstance(w, torch.Tensor) and w.ndim >= 1:
            embed_dim = int(w.shape[0])
            inferred = {768: "vit_b", 1024: "vit_l", 1280: "vit_h"}.get(embed_dim, None)
            if inferred is not None and inferred != sam_model_type:
                print(
                    json.dumps(
                        {
                            "info": "override_sam_model_type_from_checkpoint",
                            "from": sam_model_type,
                            "to": inferred,
                            "embed_dim": embed_dim,
                        },
                        ensure_ascii=False,
                    )
                )
                sam_model_type = inferred
    model = PartSegModel(
        num_classes=int(mcfg.get("num_classes", 5)),
        embedding_dim=int(mcfg.get("embedding_dim", 16)),
        feat_dim=int(mcfg.get("feat_dim", 256)),
        sam_backbone=mcfg.get("sam_backbone", "fallback"),
        dino_backbone=mcfg.get("dino_backbone", "fallback"),
        sam_ckpt=args.sam_ckpt,
        sam_model_type=sam_model_type,
    )
    try:
        model.load_state_dict(ckpt_model)
    except RuntimeError as e:
        if "sam_branch.encoder" in str(e):
            raise RuntimeError(
                "Checkpoint/model mismatch around SAM branch. "
                "If this checkpoint was trained with sam_backbone='sam', ensure segment_anything is installed "
                "and pass --sam_ckpt when needed."
            ) from e
        raise
    model.to(device).eval()

    def run_one(image_np: np.ndarray, stem: str, gt_sem=None, gt_ins=None) -> Dict[str, object]:
        image_t = torch.from_numpy(image_np).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        image_t = image_t.to(device)
        with torch.no_grad():
            out = model(image_t)
        pred_sem = out["pred_semantic"].argmax(dim=1)[0].cpu().numpy().astype(np.int64)
        pred_emb = out["pred_embedding"][0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
        pred_ins, ins_meta = _classwise_instance_grouping(
            pred_sem,
            pred_emb,
            emb_cluster_dist_th=args.emb_dist_th,
            min_instance_pixels=args.min_pixels,
            max_centers_per_class=args.max_centers_per_class,
        )

        if gt_sem is None:
            gt_sem = np.full_like(pred_sem, 255, dtype=np.int64)
        if gt_ins is None:
            gt_ins = np.full_like(pred_ins, 255, dtype=np.int64)

        vis_paths = save_infer_visuals(args.out_dir, stem, image_np, gt_sem, pred_sem, gt_ins, pred_ins)
        meta = {
            "stem": stem,
            "n_instances": len(ins_meta),
            "instances": ins_meta,
            "visualizations": vis_paths,
        }
        with open(os.path.join(args.out_dir, f"{stem}_pred.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return meta

    results: List[Dict[str, object]] = []
    if args.rgb is not None:
        image_np = _load_rgb(args.rgb)
        stem = os.path.splitext(os.path.basename(args.rgb))[0]
        gt_sem = None
        gt_ins = None
        if args.npz is not None and os.path.isfile(args.npz):
            z = np.load(args.npz)
            sem_cap = z["semantic_segmentation"].astype(np.int64)
            ins_cap = z["instance_segmentation"].astype(np.int64)
            # Keep single-image inference consistent with training supervision:
            # CAPNet semantic -> unified 5-class space, and instance filtered by valid mask.
            gt_sem = map_capnet_semantic_to_unified(sem_cap)
            valid_mask = gt_sem != IGNORE_INDEX
            gt_ins = filter_instance_with_valid_mask(ins_cap, valid_mask)
        results.append(run_one(image_np, stem, gt_sem=gt_sem, gt_ins=gt_ins))
    elif args.data_root is not None:
        ds = CapnetSingapoDataset(data_root=args.data_root, split="val")
        for i in range(len(ds)):
            item = ds[i]
            image_np = (item["image"].permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
            stem = item["meta"]["stem"]
            gt_sem = item["gt_semantic"].numpy().astype(np.int64)
            gt_ins = item["gt_instance"].numpy().astype(np.int64)
            results.append(run_one(image_np, stem, gt_sem=gt_sem, gt_ins=gt_ins))
    else:
        raise ValueError("Provide either --rgb or --data_root")

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"n_results": len(results), "results": results}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

