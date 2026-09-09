"""CTR feature-strength analysis (Q2).

The assignment is explicit: this is an *analysis* question, not a modelling one.
We rank candidate features by their statistical association with the click target
using measures that behave well under (a) severe class imbalance and (b) mixed
categorical/numeric types:

- **Weight of Evidence (WoE) + Information Value (IV)** for categorical features —
  the credit-scoring standard for ranking predictors of a rare binary event.
- **Mutual information** for numeric features — captures non-linear association
  that Pearson correlation would miss.

Crucially we compute everything on the **won** subset only, because ``conversion``
is undefined (structurally 0) for impressions we never served. We also flag
**leakage**: auction-outcome columns (bid, won_bid, feedback_bid) are not known
pre-bid and must not enter a CTR predictor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

# Fields produced by / after the auction — unavailable at the moment we'd score CTR.
LEAKAGE_COLUMNS = {"bid", "won_bid", "feedback_bid", "conversion", "clicked", "won", "clearing_price"}

# Identifiers: astronomically high cardinality, obfuscated -> not directly usable
# as features (though target-encoded aggregates could be, noted in the notebook).
IDENTIFIER_COLUMNS = {"user_id", "url", "domain", "ad_slot", "time", "user_agent", "publisher_properties"}


def woe_iv(df: pd.DataFrame, feature: str, target: str = "conversion", min_n: int = 50) -> tuple[pd.DataFrame, float]:
    """Weight of Evidence per level and total Information Value for a categorical.

    IV rule-of-thumb: <0.02 useless, 0.02-0.1 weak, 0.1-0.3 medium, >0.3 strong.
    Sparse levels (< ``min_n``) are pooled into ``__other__`` to stabilise the estimate.
    """
    s = df[[feature, target]].copy()
    counts = s[feature].value_counts()
    keep = counts[counts >= min_n].index
    s[feature] = np.where(s[feature].isin(keep), s[feature], "__other__")

    grp = s.groupby(feature)[target].agg(events="sum", n="size")
    grp["non_events"] = grp["n"] - grp["events"]
    tot_e, tot_ne = grp["events"].sum(), grp["non_events"].sum()

    # Laplace smoothing to avoid divide-by-zero on pure buckets.
    dist_e = (grp["events"] + 0.5) / (tot_e + 0.5 * len(grp))
    dist_ne = (grp["non_events"] + 0.5) / (tot_ne + 0.5 * len(grp))
    grp["woe"] = np.log(dist_e / dist_ne)
    grp["iv_contrib"] = (dist_e - dist_ne) * grp["woe"]
    grp["ctr"] = grp["events"] / grp["n"]

    iv = float(grp["iv_contrib"].sum())
    return grp.sort_values("ctr", ascending=False).round(4), iv


def rank_categorical_iv(df: pd.DataFrame, features: list[str], target: str = "conversion") -> pd.DataFrame:
    """Rank categorical candidate features by Information Value."""
    rows = []
    for f in features:
        try:
            _, iv = woe_iv(df, f, target)
            rows.append((f, iv, df[f].nunique()))
        except Exception as exc:  # keep the ranking robust to odd columns
            rows.append((f, np.nan, df[f].nunique()))
    out = pd.DataFrame(rows, columns=["feature", "information_value", "n_unique"])
    out["strength"] = pd.cut(
        out["information_value"],
        [-np.inf, 0.02, 0.1, 0.3, np.inf],
        labels=["useless", "weak", "medium", "strong"],
    )
    return out.sort_values("information_value", ascending=False).reset_index(drop=True)


def rank_numeric_mi(df: pd.DataFrame, features: list[str], target: str = "conversion", seed: int = 0) -> pd.DataFrame:
    """Rank numeric candidate features by mutual information with the target."""
    sub = df[features + [target]].dropna()
    X = sub[features].to_numpy()
    y = sub[target].astype(int).to_numpy()
    mi = mutual_info_classif(X, y, random_state=seed)
    return (
        pd.DataFrame({"feature": features, "mutual_info": mi})
        .sort_values("mutual_info", ascending=False)
        .reset_index(drop=True)
    )
