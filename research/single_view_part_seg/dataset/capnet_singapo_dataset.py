import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from research.single_view_part_seg.utils.label_mapping import (
    IGNORE_INDEX,
    filter_instance_with_valid_mask,
    map_capnet_semantic_to_unified,
)


@dataclass(frozen=True)
class SampleRecord:
    stem: str
    image_path: str
    seg_path: str


def _list_records(data_root: str, rgb_dirname: str, seg_dirname: str) -> List[SampleRecord]:
    rgb_dir = os.path.join(data_root, rgb_dirname)
    seg_dir = os.path.join(data_root, seg_dirname)
    if not os.path.isdir(rgb_dir):
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")
    if not os.path.isdir(seg_dir):
        raise FileNotFoundError(f"Segmentation directory not found: {seg_dir}")

    records: List[SampleRecord] = []
    for fn in sorted(os.listdir(rgb_dir)):
        if not fn.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        stem, _ = os.path.splitext(fn)
        image_path = os.path.join(rgb_dir, fn)
        seg_path = os.path.join(seg_dir, f"{stem}.npz")
        if not os.path.isfile(seg_path):
            continue
        records.append(SampleRecord(stem=stem, image_path=image_path, seg_path=seg_path))
    if not records:
        raise RuntimeError(f"No valid rgb/seg pairs found under: {data_root}")
    return records


def _split_records(records: List[SampleRecord], train_ratio: float, seed: int) -> Tuple[List[SampleRecord], List[SampleRecord]]:
    idx = list(range(len(records)))
    random.Random(seed).shuffle(idx)
    n_train = max(1, int(round(len(idx) * train_ratio)))
    n_train = min(n_train, len(idx) - 1) if len(idx) > 1 else 1
    train_idx = set(idx[:n_train])
    train_records = [records[i] for i in range(len(records)) if i in train_idx]
    val_records = [records[i] for i in range(len(records)) if i not in train_idx]
    if len(val_records) == 0:
        val_records = train_records[-1:]
        train_records = train_records[:-1] if len(train_records) > 1 else train_records
    return train_records, val_records


class CapnetSingapoDataset(Dataset):
    """
    Dataset output format:
    {
        "image": Tensor[3,H,W] float32 in [0,1],
        "gt_semantic": Tensor[H,W] int64 in {0,1,2,3,4,255},
        "gt_instance": Tensor[H,W] int64, ignored pixels are 255,
        "valid_mask": Tensor[H,W] bool,
        "meta": {
            "image_path": str,
            "seg_path": str,
            "orig_h": int,
            "orig_w": int,
            "stem": str
        }
    }
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        rgb_dirname: str = "rgb_512",
        seg_dirname: str = "segmentation_512",
        train_split_ratio: float = 0.9,
        seed: int = 42,
    ) -> None:
        if split not in ("train", "val"):
            raise ValueError(f"split must be train/val, got {split}")
        all_records = _list_records(data_root, rgb_dirname, seg_dirname)
        train_records, val_records = _split_records(all_records, train_split_ratio, seed)
        self.records = train_records if split == "train" else val_records
        self.split = split

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _load_image(path: str) -> np.ndarray:
        # np.asarray(PIL) may return a read-only view; copy() avoids torch warning.
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8).copy()

    @staticmethod
    def _load_seg(path: str) -> Dict[str, np.ndarray]:
        z = np.load(path)
        required = ("semantic_segmentation", "instance_segmentation")
        for k in required:
            if k not in z.files:
                raise KeyError(f"{path} missing key: {k}, available={z.files}")
        return {k: z[k] for k in z.files}

    def __getitem__(self, index: int) -> Dict[str, object]:
        rec = self.records[index]
        image_np = self._load_image(rec.image_path)  # [H,W,3], uint8
        seg = self._load_seg(rec.seg_path)

        sem_cap = seg["semantic_segmentation"].astype(np.int64)
        ins_cap = seg["instance_segmentation"].astype(np.int64)

        gt_semantic = map_capnet_semantic_to_unified(sem_cap)  # int64, {0..4,255}
        valid_mask = gt_semantic != IGNORE_INDEX               # bool
        gt_instance = filter_instance_with_valid_mask(ins_cap, valid_mask)  # int64

        image_t = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        gt_sem_t = torch.from_numpy(gt_semantic.astype(np.int64))
        gt_ins_t = torch.from_numpy(gt_instance.astype(np.int64))
        valid_t = torch.from_numpy(valid_mask.astype(np.bool_))

        sample = {
            "image": image_t,
            "gt_semantic": gt_sem_t,
            "gt_instance": gt_ins_t,
            "valid_mask": valid_t,
            "meta": {
                "image_path": rec.image_path,
                "seg_path": rec.seg_path,
                "orig_h": int(image_np.shape[0]),
                "orig_w": int(image_np.shape[1]),
                "stem": rec.stem,
            },
        }
        return sample

