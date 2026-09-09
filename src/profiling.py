"""Dataset-characterisation helpers (Q1).

Small, testable functions that return tidy DataFrames the notebook can display or
plot. Kept free of side effects (no plotting here) so they can be reused in the
report generator and unit-tested.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def schema_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column dtype, missingness and cardinality — the classic 'first look'."""
    return (
        pd.DataFrame(
            {
                "dtype": df.dtypes.astype(str),
                "n_missing": df.isna().sum(),
                "pct_missing": (df.isna().mean() * 100).round(2),
                "n_unique": df.nunique(dropna=True),
                "example": [df[c].dropna().iloc[0] if df[c].notna().any() else None for c in df.columns],
            }
        )
        .rename_axis("column")
        .reset_index()
    )


def constant_columns(df: pd.DataFrame) -> list[str]:
    """Columns with a single distinct non-null value — zero predictive value."""
    return [c for c in df.columns if df[c].nunique(dropna=True) <= 1]


def funnel(df: pd.DataFrame) -> pd.DataFrame:
    """The RTB participation funnel: requests -> bidded -> won -> clicked."""
    n = len(df)
    bidded = int(df["bidded"].sum())
    won = int((df["won_bid"] == 1).sum())
    clicked = int((df["conversion"] == 1).sum())
    rows = [
        ("Requests (opportunities)", n, 1.0, np.nan),
        ("Bidded", bidded, bidded / n, bidded / n),
        ("Won auction", won, won / n, won / bidded if bidded else np.nan),
        ("Converted (click)", clicked, clicked / n, clicked / won if won else np.nan),
    ]
    return pd.DataFrame(rows, columns=["stage", "count", "pct_of_requests", "pct_of_prev_meaningful"])


def rate_by_category(
    df: pd.DataFrame, by: str, target: str, min_n: int = 100
) -> pd.DataFrame:
    """Mean of a binary ``target`` within each level of ``by`` (with support).

    Filters out sparse levels (< ``min_n``) so noisy rates don't mislead. Used for
    win-rate and CTR breakdowns.
    """
    g = (
        df.groupby(by, dropna=False)[target]
        .agg(n="size", rate="mean")
        .reset_index()
    )
    g = g[g["n"] >= min_n].sort_values("rate", ascending=False)
    return g


def numeric_summary(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Percentile-rich describe() for skewed monetary/count columns."""
    return df[cols].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).T.round(3)
