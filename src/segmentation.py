
"""Segmentation helpers: classic RFM scoring (baseline) + matrix prep for K-Means/DBSCAN."""
import numpy as np
import pandas as pd

SEG_FEATURES = ["recency_days", "frequency", "monetary"]

# Classic RFM segment map, based on R and F scores (1-5). First match wins.
RFM_MAP = [
    (r"5[4-5]",   "Champions"),
    (r"[3-4][4-5]", "Loyal"),
    (r"[4-5][2-3]", "Potential Loyalists"),
    (r"51",       "New Customers"),
    (r"41",       "Promising"),
    (r"33",       "Need Attention"),
    (r"3[1-2]",   "About to Sleep"),
    (r"[1-2]5",   "Can't Lose"),
    (r"[1-2][3-4]", "At Risk"),
    (r"[1-2][1-2]", "Hibernating"),
]


def rfm_scores(profiles: pd.DataFrame) -> pd.DataFrame:
    """Quintile scores 1-5 (5 = best). rank(method='first') breaks ties so qcut never fails."""
    out = pd.DataFrame(index=profiles.index)
    out["R"] = pd.qcut(profiles["recency_days"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    out["F"] = pd.qcut(profiles["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    out["M"] = pd.qcut(profiles["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rf = out["R"].astype(str) + out["F"].astype(str)
    out["rfm_segment"] = "Other"
    for pattern, name in reversed(RFM_MAP):      # reversed so earlier rules win
        out.loc[rf.str.fullmatch(pattern), "rfm_segment"] = name
    return out


def seg_matrix(profiles: pd.DataFrame) -> np.ndarray:
    """Recency raw; frequency & monetary logged (Step 2 skew check). Scale AFTER this."""
    return np.column_stack([
        profiles["recency_days"].to_numpy(dtype=float),
        np.log1p(profiles["frequency"].to_numpy(dtype=float)),
        np.log1p(profiles["monetary"].to_numpy(dtype=float)),
    ])