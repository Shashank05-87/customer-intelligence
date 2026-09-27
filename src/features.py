"""Customer feature builder for the Customer Intelligence System.

Turns purchase-level rows into one profile row per customer, "as of" a snapshot date.
Only data strictly before the snapshot is used, so the same function serves both:
  - segmentation  (snapshot = end of data  -> "who is everyone now?")
  - churn model   (snapshot = cutoff date  -> no peeking into the future)
"""
import numpy as np
import pandas as pd

# columns the models will use (dates and raw counts stay out)
FEATURE_COLS = [
    "recency_days", "frequency", "monetary", "tenure_days", "avg_order_value",
    "avg_gap_days", "overdue_ratio", "spend_trend", "n_products",
    "return_rate", "is_uk", "is_one_time_buyer",
]


def build_features(tx: pd.DataFrame, cancels: pd.DataFrame, snapshot, window: int = 90) -> pd.DataFrame:
    snapshot = pd.Timestamp(snapshot)
    tx = tx[tx["InvoiceDate"] < snapshot]
    cancels = cancels[cancels["InvoiceDate"] < snapshot]
    assert tx["InvoiceDate"].max() < snapshot, "leakage: data on/after snapshot"

    g = tx.groupby("Customer ID")
    f = pd.DataFrame({
        "first_purchase": g["InvoiceDate"].min(),
        "last_purchase":  g["InvoiceDate"].max(),
        "frequency":      g["Invoice"].nunique(),     # orders, not rows
        "monetary":       g["Revenue"].sum(),
        "n_lines":        g.size(),
        "n_products":     g["StockCode"].nunique(),
        "country":        g["Country"].first(),
    })

    # --- core RFM + tenure
    f["recency_days"]    = (snapshot - f["last_purchase"]).dt.days
    f["tenure_days"]     = (snapshot - f["first_purchase"]).dt.days
    f["avg_order_value"] = f["monetary"] / f["frequency"]

    # --- personal buying rhythm (gaps between distinct purchase days)
    days = (tx.assign(day=tx["InvoiceDate"].dt.normalize())[["Customer ID", "day"]]
              .drop_duplicates()
              .sort_values(["Customer ID", "day"]))
    days["gap"] = days.groupby("Customer ID")["day"].diff().dt.days
    f["active_days"]   = days.groupby("Customer ID").size()
    f["avg_gap_days"]  = days.groupby("Customer ID")["gap"].mean()   # NaN for one-day buyers
    f["overdue_ratio"] = f["recency_days"] / f["avg_gap_days"]       # silence vs own rhythm

    # --- spend trend: last `window` days vs the `window` days before that (log ratio)
    w = pd.Timedelta(days=window)
    last = tx[tx["InvoiceDate"] >= snapshot - w]
    prev = tx[(tx["InvoiceDate"] >= snapshot - 2 * w) & (tx["InvoiceDate"] < snapshot - w)]
    f["spend_last_window"] = last.groupby("Customer ID")["Revenue"].sum()
    f["spend_prev_window"] = prev.groupby("Customer ID")["Revenue"].sum()
    f[["spend_last_window", "spend_prev_window"]] = f[["spend_last_window", "spend_prev_window"]].fillna(0)
    f["spend_trend"] = np.log1p(f["spend_last_window"]) - np.log1p(f["spend_prev_window"])

    # --- returns (from the cancellations kept aside in Step 1)
    f["n_cancel_lines"] = cancels.groupby("Customer ID").size()
    f["n_cancel_lines"] = f["n_cancel_lines"].fillna(0).astype(int)
    f["return_rate"] = f["n_cancel_lines"] / (f["n_lines"] + f["n_cancel_lines"])

    # --- flags
    f["is_uk"] = (f["country"] == "United Kingdom").astype(int)
    f["is_one_time_buyer"] = (f["active_days"] == 1).astype(int)

    return f
