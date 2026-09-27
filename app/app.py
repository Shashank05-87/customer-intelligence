
"""Customer Intelligence System — retention dashboard.
Run from the project root:  streamlit run app/app.py
"""
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
PROC, MODELS = ROOT / "data" / "processed", ROOT / "models"

st.set_page_config(page_title="Customer Intelligence", page_icon="📊", layout="wide")

SEG_ORDER = ["Champions", "Loyal Regulars", "New & Occasional", "Lapsed Regulars", "Lost One-timers"]
BAND_ORDER = ["High", "Medium", "Low", "Already lost"]
FEATURE_LABELS = {
    "recency_days": "Days since last order", "frequency": "Number of orders", "monetary": "Lifetime spend",
    "tenure_days": "Customer age (days)", "avg_order_value": "Average order value",
    "avg_gap_days": "Usual gap between orders", "overdue_ratio": "Silence vs usual rhythm",
    "spend_trend": "Recent spend trend", "n_products": "Product variety", "return_rate": "Return rate",
    "is_uk": "UK customer", "is_one_time_buyer": "One-time buyer",
}


# ---------------------------------------------------------------- data
@st.cache_data
def load():
    df = pd.read_parquet(PROC / "customers_explained.parquet").reset_index()
    df["Customer ID"] = df["Customer ID"].astype(int)
    active = df["risk_band"] != "Already lost"
    df["risk_percentile"] = np.nan
    df.loc[active, "risk_percentile"] = df.loc[active, "churn_prob"].rank(pct=True) * 100
    df["action"] = df.apply(suggest_action, axis=1)
    model = joblib.load(MODELS / "churn_model.joblib")
    base = joblib.load(MODELS / "shap_base.joblib")["base_value"]
    return df, model, base


def suggest_action(r) -> str:
    band, seg, key = r["risk_band"], r["segment"], r.get("key_account", 0) == 1
    if key and band in ("High", "Medium"):  return "Account manager call this week"
    if key and band == "Already lost":      return "Win-back call (lapsed key account)"
    if band == "Already lost":              return "Exclude from paid campaigns"
    if seg == "Champions":
        return "Personal check-in + loyalty perk" if band == "High" else "No discount needed: VIP early access"
    if seg == "Loyal Regulars":
        return {"High": "Targeted offer on favourite products", "Medium": "Reminder email"}.get(band, "Nurture towards 10+ orders")
    if seg == "New & Occasional":
        return "Second-purchase coupon" if band == "High" else "Onboarding: push towards 5th order"
    if seg == "Lapsed Regulars":            return "Win-back offer"
    if seg == "Lost One-timers":            return "Low ROI: email only, no discount"
    return "Monitor"


def gbp(x) -> str:
    return f"£{x:,.0f}"


df, model, base = load()
active = df[df["risk_band"] != "Already lost"]

# ---------------------------------------------------------------- sidebar
st.sidebar.title("📊 Customer Intelligence")
page = st.sidebar.radio("View", ["Overview", "Who to call", "Customer lookup", "Segments", "About the model"])
st.sidebar.caption(f"{len(df):,} customers · {len(active):,} active (bought in last 365 days)\n\n"
                   "Data: UCI Online Retail II (UK retailer, 2009–2011)")

# ---------------------------------------------------------------- pages
if page == "Overview":
    st.title("Customer base health")
    high = active[active["risk_band"] == "High"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active customers", f"{len(active):,}")
    c2.metric("High-risk customers", f"{len(high):,}", help="Top 25% riskiest active customers")
    c3.metric("Annual value at risk", gbp(active["value_at_risk"].sum()),
              help="Sum of churn score × last-12-month spend (estimate: scores shift with season)")
    c4.metric("Key accounts at risk", int(((df["key_account"] == 1) & df["risk_band"].isin(["High", "Medium"])).sum()))

    left, right = st.columns(2)
    with left:
        st.subheader("Revenue vs customers by segment")
        seg = df.groupby("segment").agg(customers=("Customer ID", "size"), revenue=("monetary", "sum")).reset_index()
        seg["% customers"] = 100 * seg["customers"] / seg["customers"].sum()
        seg["% revenue"] = 100 * seg["revenue"] / seg["revenue"].sum()
        long = seg.melt("segment", ["% customers", "% revenue"], var_name="measure", value_name="percent")
        st.altair_chart(alt.Chart(long).mark_bar().encode(
            y=alt.Y("segment:N", sort=SEG_ORDER, title=None), x=alt.X("percent:Q", title="%"),
            color="measure:N", yOffset="measure:N", tooltip=["segment", "measure", alt.Tooltip("percent:Q", format=".1f")]),
            width="stretch")
    with right:
        st.subheader("Risk bands")
        bands = df["risk_band"].value_counts().reindex(BAND_ORDER).fillna(0).reset_index()
        bands.columns = ["band", "customers"]
        st.altair_chart(alt.Chart(bands).mark_bar().encode(
            x=alt.X("band:N", sort=BAND_ORDER, title=None), y="customers:Q",
            color=alt.Color("band:N", sort=BAND_ORDER, legend=None,
                            scale=alt.Scale(domain=BAND_ORDER, range=["#d62728", "#ff7f0e", "#2ca02c", "#9e9e9e"])),
            tooltip=["band", "customers"]), width="stretch")

    st.subheader("What the model learned")
    st.info("**~100 days of silence** is the tipping point: contact customers at 80–90 days.  \n"
            "**Twice their usual gap** between orders means a customer is drifting.  \n"
            "**The 5th order** is the loyalty threshold: below it, customers are likely to leave.")

elif page == "Who to call":
    st.title("Who to call")
    st.caption("Riskiest customers first filtered by band, then sorted by value at risk, so safe big accounts don't crowd the list.")
    f1, f2, f3 = st.columns([2, 3, 1])
    bands = f1.multiselect("Risk band", ["High", "Medium", "Low"], default=["High"])
    segs = f2.multiselect("Segment", SEG_ORDER, default=[s for s in SEG_ORDER if s in df["segment"].unique()])
    key_only = f3.toggle("Key accounts only")

    view = active[active["risk_band"].isin(bands) & active["segment"].isin(segs)]
    if key_only:
        view = view[view["key_account"] == 1]
    view = view.sort_values("value_at_risk", ascending=False)

    cols = ["Customer ID", "segment", "frequency", "recency_days", "annual_value", "churn_prob",
            "value_at_risk", "reason_1", "action"]
    st.write(f"**{len(view):,} customers** · {gbp(view['value_at_risk'].sum())} annual value at risk")
    st.dataframe(view[cols], hide_index=True, width="stretch", column_config={
        "segment": "Segment", "frequency": "Orders", "recency_days": "Days silent",
        "annual_value": st.column_config.NumberColumn("Last-12m spend", format="£%.0f"),
        "churn_prob": st.column_config.ProgressColumn("Risk score", min_value=0, max_value=1, format="percent"),
        "value_at_risk": st.column_config.NumberColumn("Value at risk", format="£%.0f"),
        "reason_1": "Main reason", "action": "Suggested action"})
    st.download_button("⬇️ Download list (CSV)", view[cols].to_csv(index=False).encode(),
                       "call_list.csv", "text/csv")

elif page == "Customer lookup":
    st.title("Customer lookup")
    default = active.sort_values("value_at_risk", ascending=False)["Customer ID"].iloc[0]
    ids = df["Customer ID"].sort_values().tolist()
    cid = st.selectbox("Customer ID", ids, index=ids.index(default))
    r = df[df["Customer ID"] == cid].iloc[0]

    st.subheader(f"Customer {cid} · {r['segment']}" + (" · 🔑 Key account" if r["key_account"] == 1 else ""))
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Orders", f"{r['frequency']:.0f}")
    c2.metric("Days since last order", f"{r['recency_days']:.0f}")
    c3.metric("Lifetime spend", gbp(r["monetary"]))
    c4.metric("Last-12m spend", gbp(r["annual_value"]))
    c5.metric("Risk band", r["risk_band"])

    if r["risk_band"] == "Already lost":
        st.warning("No purchase in over a year: treated as already lost, so no risk score is given.")
    else:
        st.write(f"**Risk score {r['churn_prob']:.0%}**, riskier than {r['risk_percentile']:.0f}% of active customers. "
                 f"Value at risk: **{gbp(r['value_at_risk'])}**/year.")
        st.markdown("**Why:**\n" + "\n".join(f"- {r[f'reason_{i}']}" for i in (1, 2, 3)))

        contrib = pd.DataFrame({"feature": [FEATURE_LABELS.get(f, f) for f in model["features"]],
                                "points": [r[f"shap_{f}"] * 100 for f in model["features"]]})
        contrib["effect"] = np.where(contrib["points"] > 0, "raises risk", "lowers risk")
        st.altair_chart(alt.Chart(contrib).mark_bar().encode(
            x=alt.X("points:Q", title=f"contribution (pts) · starts from {base:.0%} average"),
            y=alt.Y("feature:N", sort=alt.EncodingSortField("points", op="sum", order="descending"), title=None),
            color=alt.Color("effect:N", scale=alt.Scale(domain=["raises risk", "lowers risk"],
                                                        range=["#d62728", "#1f77b4"])),
            tooltip=["feature", alt.Tooltip("points:Q", format="+.1f")]), width="stretch")
    st.success(f"**Suggested action:** {r['action']}")

elif page == "Segments":
    st.title("Segments")
    summary = df.groupby("segment").agg(
        customers=("Customer ID", "size"), median_orders=("frequency", "median"),
        median_days_silent=("recency_days", "median"), median_spend=("monetary", "median"),
        revenue=("monetary", "sum"), avg_risk=("churn_prob", "mean"), key_accounts=("key_account", "sum"))
    summary["% revenue"] = 100 * summary["revenue"] / summary["revenue"].sum()
    summary = summary.reindex([s for s in SEG_ORDER if s in summary.index]).drop(columns="revenue")
    st.dataframe(summary, width="stretch", column_config={
        "median_spend": st.column_config.NumberColumn("Median spend", format="£%.0f"),
        "avg_risk": st.column_config.NumberColumn("Avg risk (active)", format="%.2f"),
        "% revenue": st.column_config.NumberColumn(format="%.1f%%")})
    st.markdown("""
| Segment | Who they are | Default play |
|---|---|---|
| **Champions** | Frequent, recent, high spend | Protect: no discounts needed; VIP perks |
| **Loyal Regulars** | Steady repeat buyers | Nurture; targeted offers if risk rises |
| **New & Occasional** | 1–2 recent orders | Get them to their 5th order |
| **Lapsed Regulars** | Used to buy repeatedly, silent ~1 year | Win-back offer |
| **Lost One-timers** | One small order, long gone | Low ROI: email only |

Segments come from K-Means on recency, log(orders), log(spend). They describe *who* a customer is;
the risk score describes *whether they are leaving*. Check the order count too: a single large order can land in "Loyal Regulars".
""")

else:  # About the model
    st.title("About the model")
    st.markdown(f"""
**Question:** will an active customer make *no* purchase in the next {model['horizon']} days?

**Design**
- Time-based snapshots: features use only data before each cutoff; labels come from the {model['horizon']} days after.
- Trained on 4 past cutoffs, tuned on the latest of them, tested once on the final cutoff.
- Only customers active in the prior {model['active_window']} days are scored.
- Chosen model: **{model['family']}** (selected on validation, not test).
""")
    st.subheader("Test results")
    st.dataframe(model["test_report"].style.format("{:.3f}"), width="stretch")
    st.markdown("""
**Limitations**
- **Seasonality:** churn is ~15–20 pts lower in windows covering the Christmas rush. With only one prior season,
  probabilities can't be calibrated reliably, so risk bands are **rank-based** (top 25% = High).
- **Partial returns** aren't netted against their original orders, which slightly inflates some spend.
- **No campaign data:** suggested actions are rules, not proven effects. Next step would be an A/B test with a holdout group.
- Data is 2009–2011, mostly UK wholesale-leaning customers.
""")