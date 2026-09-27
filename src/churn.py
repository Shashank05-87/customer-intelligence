"""Churn labelling: time-based snapshots so the model never sees the future.

For a cutoff date T:
  features = customer profile built ONLY from purchases before T   (src/features.py)
  label    = 1 if the customer makes NO purchase in [T, T + horizon)
Only customers active in the `active_window` days before T are included;
customers gone for a year+ are already lost, and predicting that is trivially easy.

Seasonality: windows covering the pre-Christmas rush have ~20pt lower churn.
`holiday_window` tells the model which kind of window it is predicting for.
"""
import pandas as pd
from features import build_features, FEATURE_COLS

MODEL_COLS = FEATURE_COLS   # holiday_window tested in 4.10b: overcorrected -> not used

def holiday_flag(cutoff, horizon=90, min_overlap_days=15) -> int:
    """1 if the prediction window covers at least `min_overlap_days` of November (peak pre-Christmas buying)."""
    cutoff = pd.Timestamp(cutoff)
    end = cutoff + pd.Timedelta(days=horizon)
    for year in (cutoff.year, cutoff.year + 1):
        nov_start, nov_end = pd.Timestamp(year, 11, 1), pd.Timestamp(year, 12, 1)
        overlap = (min(end, nov_end) - max(cutoff, nov_start)).days
        if overlap >= min_overlap_days:
            return 1
    return 0


def make_snapshot(tx, cancels, cutoff, horizon=90, active_window=365):
    cutoff = pd.Timestamp(cutoff)
    f = build_features(tx, cancels, cutoff)
    f = f[f["recency_days"] <= active_window].copy()

    window = tx[(tx["InvoiceDate"] >= cutoff) &
                (tx["InvoiceDate"] < cutoff + pd.Timedelta(days=horizon))]
    returned = set(window["Customer ID"].unique())
    f["churned"] = (~f.index.isin(returned)).astype(int)
    f["holiday_window"] = holiday_flag(cutoff, horizon)
    f["cutoff"] = cutoff
    return f