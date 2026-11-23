"""Metrics and result files shared by all model notebooks.

Every model calls `save_results()` once with its test-set predictions. Notebook 07
then reads everything back with `load_all_results()`.
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import paths


def compute_metrics(y_true, y_pred) -> dict:
    """RMSE and MAE in °C, R², Spearman correlation, number of test rows."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    # Spearman is undefined for a constant prediction (e.g. the mean baseline).
    spearman = None if np.ptp(y_pred) == 0 else float(spearmanr(y_true, y_pred).statistic)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": spearman,
        "n_test": int(len(y_true)),
    }


def save_results(model_name: str, y_test: pd.Series, y_pred, params: dict | None = None) -> dict:
    """Write results/metrics/<model>.json and results/predictions/<model>.csv. Returns the metrics."""
    metrics = compute_metrics(y_test, y_pred)
    record = {
        "model": model_name,
        **metrics,
        "params": params or {},
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    paths.METRICS.mkdir(parents=True, exist_ok=True)
    paths.PREDICTIONS.mkdir(parents=True, exist_ok=True)
    (paths.METRICS / f"{model_name}.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame({"id": y_test.index, "tm_true": y_test.to_numpy(), "tm_pred": np.asarray(y_pred)}).to_csv(
        paths.PREDICTIONS / f"{model_name}.csv", index=False
    )
    return metrics


def load_all_results() -> pd.DataFrame:
    """One row per model, sorted by RMSE."""
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(paths.METRICS.glob("*.json"))]
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values("rmse", ignore_index=True)
