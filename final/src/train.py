from __future__ import annotations
import argparse
import os
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MRHiSumDataset, load_mr_hisum, make_synthetic_dataset, split_items, collate_variable
from metrics import regression_metrics, highlight_metrics
from model import TemporalMLP, BiLSTMRegressor
from utils import set_seed, ensure_dir, get_device


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate(model, loader, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            for item in batch:
                x = item["features"].to(device)
                y = item["gtscore"].cpu().numpy()
                pred = model(x).detach().cpu().numpy()
                m = regression_metrics(y, pred)
                m.update(highlight_metrics(y, pred, 0.15))
                m.update(highlight_metrics(y, pred, 0.50))
                m["video_id"] = item["video_id"]
                rows.append(m)
    df = pd.DataFrame(rows)
    return df, df.drop(columns=["video_id"]).mean(numeric_only=True).to_dict()


def train_epoch(model, loader, optimizer, device, grad_clip):
    model.train()
    loss_fn = torch.nn.MSELoss()
    losses = []
    for batch in tqdm(loader, desc="train", leave=False):
        optimizer.zero_grad()
        total = 0
        for item in batch:
            x = item["features"].to(device)
            y = item["gtscore"].to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            total = total + loss
        total = total / len(batch)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(float(total.detach().cpu()))
    return float(np.mean(losses))


def main(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    ensure_dir(cfg["outputs"]["dir"])
    ensure_dir(os.path.join(cfg["outputs"]["dir"], "tables"))
    ensure_dir(cfg["outputs"]["checkpoints_dir"])

    try:
        items = load_mr_hisum(cfg["data"]["h5_path"], cfg["data"].get("max_videos"))
        data_mode = "mr_hisum"
    except FileNotFoundError:
        if not cfg["data"].get("use_synthetic_if_missing", True):
            raise
        items = make_synthetic_dataset(seed=cfg["seed"])
        data_mode = "synthetic_demo"

    train_items, val_items, test_items = split_items(items, cfg["data"]["val_size"], cfg["data"]["test_size"], cfg["seed"])
    train_loader = DataLoader(MRHiSumDataset(train_items), batch_size=cfg["training"]["batch_size"], shuffle=True, collate_fn=collate_variable)
    val_loader = DataLoader(MRHiSumDataset(val_items), batch_size=cfg["training"]["batch_size"], shuffle=False, collate_fn=collate_variable)
    test_loader = DataLoader(MRHiSumDataset(test_items), batch_size=cfg["training"]["batch_size"], shuffle=False, collate_fn=collate_variable)

    input_dim = train_items[0].features.shape[1]
    if cfg["model"].get("type") == "bilstm":
        model = BiLSTMRegressor(input_dim=input_dim, hidden_dim=cfg["model"]["hidden_dim"] // 2, dropout=cfg["model"]["dropout"])
    else:
        model = TemporalMLP(input_dim=input_dim, hidden_dim=cfg["model"]["hidden_dim"], dropout=cfg["model"]["dropout"])
    device = get_device()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    history = []
    best_val = float("inf")
    bad_epochs = 0
    ckpt_path = os.path.join(cfg["outputs"]["checkpoints_dir"], "best_model.pt")

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, cfg["training"]["grad_clip"])
        _, val_metrics = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}, "data_mode": data_mode}
        history.append(row)
        print(row)
        if val_metrics["mse"] < best_val:
            best_val = val_metrics["mse"]
            bad_epochs = 0
            torch.save({"model_state": model.state_dict(), "input_dim": input_dim, "config": cfg}, ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["training"]["patience"]:
                print("Early stopping")
                break

    pd.DataFrame(history).to_csv(os.path.join(cfg["outputs"]["dir"], "tables", "training_history.csv"), index=False)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_per_video, test_avg = evaluate(model, test_loader, device)
    test_per_video.to_csv(os.path.join(cfg["outputs"]["dir"], "tables", "test_per_video_metrics.csv"), index=False)
    pd.DataFrame([{**test_avg, "data_mode": data_mode, "n_train": len(train_items), "n_val": len(val_items), "n_test": len(test_items)}]).to_csv(os.path.join(cfg["outputs"]["dir"], "tables", "test_metrics_summary.csv"), index=False)
    print("Final test metrics:", test_avg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
