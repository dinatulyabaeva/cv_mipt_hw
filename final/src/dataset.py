from __future__ import annotations
import os
from dataclasses import dataclass
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class VideoItem:
    video_id: str
    features: np.ndarray
    gtscore: np.ndarray


class MRHiSumDataset(Dataset):
    def __init__(self, items: list[VideoItem]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        return {
            "video_id": item.video_id,
            "features": torch.tensor(item.features, dtype=torch.float32),
            "gtscore": torch.tensor(item.gtscore, dtype=torch.float32),
        }


def load_mr_hisum(h5_path: str, max_videos: int | None = None) -> list[VideoItem]:
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"H5 file not found: {h5_path}")
    items: list[VideoItem] = []
    with h5py.File(h5_path, "r") as h5:
        keys = list(h5.keys())
        if max_videos is not None:
            keys = keys[:max_videos]
        for key in keys:
            group = h5[key]
            if "features" not in group or "gtscore" not in group:
                continue
            features = np.asarray(group["features"])
            score = np.asarray(group["gtscore"]).reshape(-1)
            # Align lengths conservatively.
            n = min(len(features), len(score))
            if n < 2:
                continue
            features = features[:n]
            score = score[:n]
            if score.max() > score.min():
                score = (score - score.min()) / (score.max() - score.min())
            items.append(VideoItem(video_id=str(key), features=features, gtscore=score.astype("float32")))
    return items


def make_synthetic_dataset(n_videos: int = 12, feature_dim: int = 32, seed: int = 42) -> list[VideoItem]:
    """Demo fallback. It is not a replacement for MR.HiSum, but lets the pipeline run anywhere."""
    rng = np.random.default_rng(seed)
    items = []
    for i in range(n_videos):
        length = int(rng.integers(60, 140))
        x = rng.normal(0, 1, size=(length, feature_dim)).astype("float32")
        t = np.linspace(0, 1, length)
        peaks = np.zeros(length)
        for _ in range(int(rng.integers(1, 4))):
            center = rng.uniform(0.1, 0.9)
            width = rng.uniform(0.025, 0.08)
            peaks += np.exp(-((t - center) ** 2) / (2 * width ** 2))
        signal = peaks + 0.08 * rng.normal(size=length)
        signal = (signal - signal.min()) / (signal.max() - signal.min() + 1e-8)
        # Inject weak correlation into the first feature block.
        x[:, :16] += signal[:, None] * 0.8
        items.append(VideoItem(video_id=f"synthetic_{i:04d}", features=x, gtscore=signal.astype("float32")))
    return items


def split_items(items: list[VideoItem], val_size: float, test_size: float, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(items))
    rng.shuffle(idx)
    n_test = int(round(len(items) * test_size))
    n_val = int(round(len(items) * val_size))
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]
    return ([items[i] for i in train_idx], [items[i] for i in val_idx], [items[i] for i in test_idx])


def collate_variable(batch):
    return batch
