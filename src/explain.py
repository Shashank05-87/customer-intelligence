"""Turn SHAP values into plain-English reasons a manager can read."""
import numpy as np
import pandas as pd


def describe(feature: str, row: pd.Series) -> str:
    v = row[feature]
    missing = pd.isna(v)
    if feature == "recency_days":      return f"Last order {v:.0f} day{'s' if v != 1 else ''} ago"
    if feature == "frequency":         return f"{v:.0f} order{'s' if v != 1 else ''} in total"
    if feature == "monetary":          return f"£{v:,.0f} lifetime spend"
    if feature == "tenure_days":       return f"Customer for {v:.0f} days"
    if feature == "avg_order_value":   return f"Average order £{v:,.0f}"
    if feature == "avg_gap_days":      return "Only one purchase day so far" if missing else f"Usually orders every {v:.0f} days"
    if feature == "overdue_ratio":
        if missing:                    return "No buying rhythm yet (one purchase day)"
        return f"Within their usual buying rhythm ({v:.1f}x gap)" if v < 1 else f"Silent for {v:.1f}x their usual gap"
    if feature == "spend_trend":
        last, prev = row.get("spend_last_window", np.nan), row.get("spend_prev_window", np.nan)
        if last == 0 and prev == 0:    return "No purchases in the last 6 months"
        if last == 0:                  return "No spend in the last 90 days"
        return "Spending up vs previous 90 days" if v > 0 else "Spending down vs previous 90 days"
    if feature == "n_products":        return f"Bought {v:.0f} different products"
    if feature == "return_rate":       return f"{v:.0%} of order lines returned"
    if feature == "is_uk":             return "UK customer" if v == 1 else "International customer"
    if feature == "is_one_time_buyer": return "Has bought on only one day" if v == 1 else "Repeat buyer"
    return f"{feature} = {v}"


def top_reasons(shap_row: pd.Series, feature_row: pd.Series, n: int = 3) -> list[str]:
    """n biggest drivers (either direction), e.g. 'Last order 235 days ago -> raises risk (+18 pts)'."""
    out = []
    for feat in shap_row.abs().sort_values(ascending=False).index[:n]:
        s = shap_row[feat]
        direction = "raises" if s > 0 else "lowers"
        out.append(f"{describe(feat, feature_row)} -> {direction} risk ({s * 100:+.0f} pts)")
    return out