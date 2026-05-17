from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import mean_squared_error, mean_absolute_error, average_precision_score, precision_score, recall_score, f1_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    sp = spearmanr(y_true, y_pred).correlation if len(y_true) > 1 else np.nan
    kt = kendalltau(y_true, y_pred).correlation if len(y_true) > 1 else np.nan
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "spearman": float(0.0 if np.isnan(sp) else sp),
        "kendall": float(0.0 if np.isnan(kt) else kt),
    }


def topk_binary(scores: np.ndarray, ratio: float = 0.15) -> np.ndarray:
    scores = np.asarray(scores).reshape(-1)
    k = max(1, int(round(len(scores) * ratio)))
    idx = np.argsort(scores)[-k:]
    out = np.zeros_like(scores, dtype=int)
    out[idx] = 1
    return out


def highlight_metrics(y_true_score: np.ndarray, y_pred_score: np.ndarray, ratio: float = 0.15) -> dict:
    y_true_bin = topk_binary(y_true_score, ratio)
    y_pred_bin = topk_binary(y_pred_score, ratio)
    try:
        ap = average_precision_score(y_true_bin, y_pred_score)
    except Exception:
        ap = 0.0
    return {
        f"map@{int(ratio * 100)}": float(ap),
        f"precision@{int(ratio * 100)}": float(precision_score(y_true_bin, y_pred_bin, zero_division=0)),
        f"recall@{int(ratio * 100)}": float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
        f"f1@{int(ratio * 100)}": float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
    }
