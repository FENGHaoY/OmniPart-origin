import argparse
import json
import os
from typing import Dict

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from research.single_view_part_seg.dataset.capnet_singapo_dataset import CapnetSingapoDataset
from research.single_view_part_seg.losses.embedding_loss import DiscriminativeEmbeddingLoss
from research.single_view_part_seg.losses.semantic_loss import SemanticLoss
from research.single_view_part_seg.models.part_seg_model import PartSegModel
from research.single_view_part_seg.utils.config import Config
from research.single_view_part_seg.utils.metrics import metric_dict


def _safe_torch_save(obj: object, path: str) -> tuple[bool, str]:
    """
    Atomic-ish checkpoint save:
    1) save to temp file in same directory
    2) replace target file
    Returns (ok, message).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
        return True, "ok"
    except Exception as e:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return False, str(e)


def _set_requires_grad(module: torch.nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


def _count_params(module: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def _build_loaders(cfg: Config) -> tuple[DataLoader, DataLoader]:
    train_ds = CapnetSingapoDataset(
        data_root=cfg.dataset.data_root,
        split="train",
        rgb_dirname=cfg.dataset.rgb_dirname,
        seg_dirname=cfg.dataset.seg_dirname,
        train_split_ratio=cfg.dataset.train_split_ratio,
        seed=cfg.dataset.seed,
    )
    val_ds = CapnetSingapoDataset(
        data_root=cfg.dataset.data_root,
        split="val",
        rgb_dirname=cfg.dataset.rgb_dirname,
        seg_dirname=cfg.dataset.seg_dirname,
        train_split_ratio=cfg.dataset.train_split_ratio,
        seed=cfg.dataset.seed,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def _to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "image": batch["image"].to(device),
        "gt_semantic": batch["gt_semantic"].to(device),
        "gt_instance": batch["gt_instance"].to(device),
        "valid_mask": batch["valid_mask"].to(device),
    }


def _epoch_step(
    model: PartSegModel,
    loader: DataLoader,
    sem_loss_fn: SemanticLoss,
    emb_loss_fn: DiscriminativeEmbeddingLoss,
    alpha_emb: float,
    device: torch.device,
    optimizer=None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_sem = 0.0
    total_emb = 0.0
    total_miou = 0.0
    total_acc = 0.0
    steps = 0

    pbar = tqdm(loader, desc="train" if is_train else "val", leave=False)
    for batch in pbar:
        bt = _to_device(batch, device)
        out = model(bt["image"])
        pred_sem = out["pred_semantic"]
        pred_emb = out["pred_embedding"]

        l_sem = sem_loss_fn(pred_sem, bt["gt_semantic"])
        l_emb = emb_loss_fn(pred_emb, bt["gt_instance"], bt["valid_mask"])
        loss = l_sem + alpha_emb * l_emb

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            pred_label = pred_sem.argmax(dim=1)
            m = metric_dict(pred_label, bt["gt_semantic"], ignore_index=255)

        total_loss += float(loss.item())
        total_sem += float(l_sem.item())
        total_emb += float(l_emb.item())
        total_miou += m["miou"]
        total_acc += m["pixel_acc"]
        steps += 1

        pbar.set_postfix(
            loss=f"{total_loss/steps:.4f}",
            sem=f"{total_sem/steps:.4f}",
            emb=f"{total_emb/steps:.4f}",
            miou=f"{total_miou/steps:.4f}",
            acc=f"{total_acc/steps:.4f}",
        )

    if steps == 0:
        return {"loss": 0.0, "sem_loss": 0.0, "emb_loss": 0.0, "miou": 0.0, "pixel_acc": 0.0}
    return {
        "loss": total_loss / steps,
        "sem_loss": total_sem / steps,
        "emb_loss": total_emb / steps,
        "miou": total_miou / steps,
        "pixel_acc": total_acc / steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="research/outputs/capnet_all_512")
    parser.add_argument("--output_dir", default="research/outputs/single_view_part_seg")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for embedding loss")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sam_backbone", default="fallback", choices=["fallback", "sam"])
    parser.add_argument("--sam_model_type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--dino_backbone", default="fallback", choices=["fallback", "dinov2", "timm_resnet18"])
    parser.add_argument("--sam_ckpt", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--freeze_sam", action="store_true", default=True, help="Freeze SAM branch encoder.")
    parser.add_argument("--freeze_dino", action="store_true", default=True, help="Freeze DINO branch encoder.")
    parser.add_argument("--unfreeze_sam", action="store_true", help="Override and unfreeze SAM encoder.")
    parser.add_argument("--unfreeze_dino", action="store_true", help="Override and unfreeze DINO encoder.")
    parser.add_argument("--save_every", type=int, default=1, help="Save full checkpoint every N epochs.")
    args = parser.parse_args()

    cfg = Config()
    cfg.dataset.data_root = args.data_root
    cfg.train.output_dir = args.output_dir
    cfg.train.num_epochs = args.epochs
    cfg.train.batch_size = args.batch_size
    cfg.train.lr = args.lr
    cfg.loss.alpha_emb = args.alpha
    cfg.train.device = args.device
    cfg.model.sam_backbone = args.sam_backbone
    cfg.model.dino_backbone = args.dino_backbone
    cfg.train.resume_ckpt = args.resume
    cfg.train.save_every = int(args.save_every)

    os.makedirs(cfg.train.output_dir, exist_ok=True)
    with open(os.path.join(cfg.train.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "dataset": cfg.dataset.__dict__,
            "model": cfg.model.__dict__,
            "loss": cfg.loss.__dict__,
            "train": cfg.train.__dict__,
        }, f, ensure_ascii=False, indent=2)

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = _build_loaders(cfg)

    model = PartSegModel(
        num_classes=cfg.model.num_classes,
        embedding_dim=cfg.model.embedding_dim,
        feat_dim=cfg.model.feat_dim,
        sam_backbone=cfg.model.sam_backbone,
        dino_backbone=cfg.model.dino_backbone,
        sam_ckpt=args.sam_ckpt,
        sam_model_type=args.sam_model_type,
    ).to(device)

    # By default SAM/DINO are frozen feature extractors.
    freeze_sam = bool(args.freeze_sam and not args.unfreeze_sam)
    freeze_dino = bool(args.freeze_dino and not args.unfreeze_dino)
    _set_requires_grad(model.sam_branch.encoder, not freeze_sam)
    _set_requires_grad(model.dino_branch.encoder, not freeze_dino)

    total_params, trainable_params = _count_params(model)
    print(
        json.dumps(
            {
                "freeze_sam": freeze_sam,
                "freeze_dino": freeze_dino,
                "total_params": total_params,
                "trainable_params": trainable_params,
            },
            ensure_ascii=False,
        )
    )

    sem_loss_fn = SemanticLoss(ignore_index=cfg.loss.ignore_index, use_dice=cfg.loss.use_dice).to(device)
    emb_loss_fn = DiscriminativeEmbeddingLoss(
        delta_var=cfg.loss.emb_delta_var,
        delta_dist=cfg.loss.emb_delta_dist,
        pull_weight=cfg.loss.emb_pull_weight,
        push_weight=cfg.loss.emb_push_weight,
        reg_weight=cfg.loss.emb_reg_weight,
        ignore_index=cfg.loss.ignore_index,
    ).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.train.num_epochs)
    start_epoch = 0

    if cfg.train.resume_ckpt:
        ckpt = torch.load(cfg.train.resume_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt["epoch"]) + 1

    log_path = os.path.join(cfg.train.output_dir, "train_log.jsonl")
    best_val_miou = -1.0

    for epoch in range(start_epoch, cfg.train.num_epochs):
        train_stat = _epoch_step(
            model, train_loader, sem_loss_fn, emb_loss_fn, cfg.loss.alpha_emb, device, optimizer=optimizer
        )
        val_stat = _epoch_step(
            model, val_loader, sem_loss_fn, emb_loss_fn, cfg.loss.alpha_emb, device, optimizer=None
        )
        scheduler.step()

        summary = {
            "epoch": epoch,
            "train": train_stat,
            "val": val_stat,
            "lr": optimizer.param_groups[0]["lr"],
        }
        print(json.dumps(summary, ensure_ascii=False))
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": {
                "dataset": cfg.dataset.__dict__,
                "model": cfg.model.__dict__,
                "loss": cfg.loss.__dict__,
                "train": cfg.train.__dict__,
            },
        }

        if (epoch + 1) % max(1, int(cfg.train.save_every)) == 0:
            ckpt_path = os.path.join(cfg.train.output_dir, f"ckpt_epoch_{epoch:03d}.pt")
            ok, msg = _safe_torch_save(ckpt, ckpt_path)
            if not ok:
                print(json.dumps({"warn": "checkpoint_save_failed", "path": ckpt_path, "error": msg}, ensure_ascii=False))
                # fallback: save model-only state (smaller file)
                model_only = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "config": ckpt["config"],
                }
                fallback_path = os.path.join(cfg.train.output_dir, f"ckpt_epoch_{epoch:03d}_model_only.pt")
                ok2, msg2 = _safe_torch_save(model_only, fallback_path)
                if not ok2:
                    print(
                        json.dumps(
                            {"warn": "model_only_save_failed", "path": fallback_path, "error": msg2},
                            ensure_ascii=False,
                        )
                    )

        if val_stat["miou"] > best_val_miou:
            best_val_miou = val_stat["miou"]
            best_path = os.path.join(cfg.train.output_dir, "best.pt")
            ok, msg = _safe_torch_save(ckpt, best_path)
            if not ok:
                print(json.dumps({"warn": "best_save_failed", "path": best_path, "error": msg}, ensure_ascii=False))
                best_model_only = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "config": ckpt["config"],
                }
                _safe_torch_save(best_model_only, os.path.join(cfg.train.output_dir, "best_model_only.pt"))


if __name__ == "__main__":
    main()

