from __future__ import annotations
import argparse
import os
import yaml
import h5py
import numpy as np
import pandas as pd
import torch
from model import TemporalMLP, BiLSTMRegressor
from metrics import topk_binary


def build_model(checkpoint, device):
    cfg = checkpoint["config"]
    input_dim = checkpoint["input_dim"]
    if cfg["model"].get("type") == "bilstm":
        model = BiLSTMRegressor(input_dim=input_dim, hidden_dim=cfg["model"]["hidden_dim"] // 2, dropout=cfg["model"]["dropout"])
    else:
        model = TemporalMLP(input_dim=input_dim, hidden_dim=cfg["model"]["hidden_dim"], dropout=cfg["model"]["dropout"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def main(ckpt_path: str, h5_path: str, video_id: str, out_csv: str, ratio: float):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = build_model(checkpoint, device)
    with h5py.File(h5_path, "r") as h5:
        if video_id not in h5:
            raise KeyError(f"Video id {video_id} not found. Example keys: {list(h5.keys())[:5]}")
        features = np.asarray(h5[video_id]["features"])
    with torch.no_grad():
        pred = model(torch.tensor(features, dtype=torch.float32, device=device)).cpu().numpy()
    highlights = topk_binary(pred, ratio)
    df = pd.DataFrame({"segment_idx": np.arange(len(pred)), "predicted_gtscore": pred, "is_highlight": highlights})
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved inference result to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--h5", default="data/mr_hisum.h5")
    parser.add_argument("--video_id", required=True)
    parser.add_argument("--out_csv", default="outputs/tables/inference_result.csv")
    parser.add_argument("--ratio", type=float, default=0.15)
    args = parser.parse_args()
    main(args.checkpoint, args.h5, args.video_id, args.out_csv, args.ratio)
