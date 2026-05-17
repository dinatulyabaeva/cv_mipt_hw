from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)

# Demo charts for report template. Replace with real outputs after training on MR.HiSum.
epochs = np.arange(1, 16)
train = 0.092 * np.exp(-epochs / 8) + 0.018
val = 0.088 * np.exp(-epochs / 7) + 0.024 + np.random.default_rng(42).normal(0, 0.0015, len(epochs))
pd.DataFrame({"epoch": epochs, "train_loss": train, "val_mse": val}).to_csv("outputs/tables/example_training_history.csv", index=False)
plt.figure(figsize=(7, 4))
plt.plot(epochs, train, label="train MSE")
plt.plot(epochs, val, label="validation MSE")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training dynamics template")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/figures/training_curve.png", dpi=200)
plt.close()

rng = np.random.default_rng(42)
gt = np.clip(rng.beta(2, 6, 1000), 0, 1)
plt.figure(figsize=(7, 4))
plt.hist(gt, bins=30)
plt.xlabel("gtscore")
plt.ylabel("count")
plt.title("Target distribution template")
plt.tight_layout()
plt.savefig("outputs/figures/gtscore_distribution.png", dpi=200)
plt.close()

x = np.linspace(0, 1, 120)
true = np.exp(-((x - .35) ** 2) / .01) + 0.7 * np.exp(-((x - .72) ** 2) / .004) + 0.03*rng.normal(size=120)
true = (true - true.min()) / (true.max() - true.min())
pred = np.convolve(true, np.ones(7)/7, mode="same") + 0.05*rng.normal(size=120)
pred = np.clip(pred, 0, 1)
plt.figure(figsize=(8, 4))
plt.plot(true, label="target gtscore")
plt.plot(pred, label="predicted score")
plt.xlabel("segment")
plt.ylabel("score")
plt.title("Prediction vs target template")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/figures/prediction_vs_target.png", dpi=200)
plt.close()

metrics = pd.DataFrame([
    {"model": "Mean baseline", "MSE": 0.071, "MAE": 0.208, "Spearman": 0.000, "mAP@15": 0.150},
    {"model": "Ridge baseline", "MSE": 0.058, "MAE": 0.181, "Spearman": 0.180, "mAP@15": 0.235},
    {"model": "Temporal MLP", "MSE": 0.044, "MAE": 0.151, "Spearman": 0.330, "mAP@15": 0.370},
])
metrics.to_csv("outputs/tables/example_metrics_table.csv", index=False)
