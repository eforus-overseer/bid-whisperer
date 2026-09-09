"""Data loading & schema hygiene for the RTB impression dataset.

The raw file is a single-day (2021-02-01), US-only subset of ad-request/impression
logs. Every row is one auction *opportunity*; whether we participated, won, and
converted are recorded downstream in the same row.

Key modelling nuance encoded here
----------------------------------
`bid`, `won_bid`, `feedback_bid` and `conversion` are *only defined when we bid*
(``bidded == 1``). They are NaN for the ~50% of rows we skipped. We therefore keep
NaN as a meaningful "not applicable" marker rather than imputing it away.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

# Columns whose stored dtype needs pinning to avoid pandas' mixed-type warnings
# (postal_code arrives as a mix of ints, floats and strings across chunks).
_DTYPE_OVERRIDES = {"postal_code": "string"}

# Sentinel values the provider uses for "could not estimate".
SENTINELS = {"viewability": -1, "session_depth": -1}

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "data.csv"


def load_raw(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the CSV with correct dtypes and a parsed UTC timestamp.

    Adds an ``hour`` column for intraday analysis. Does not drop or impute
    anything; that is the caller's decision, made explicit in the notebook.
    """
    df = pd.read_csv(path, dtype=_DTYPE_OVERRIDES, low_memory=False)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df["hour"] = df["time"].dt.hour
    return df


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Attach analysis-friendly derived columns.

    - ``sentinel``-aware "missing" flags for viewability / session_depth.
    - ``clearing_price``: the market price we had to *beat* to win the slot. This is
      the backbone of the Q3 bid-landscape model. Under both auction types the
      ``feedback_bid`` is the price that decided the auction (winner's bid when we
      lost; 2nd-highest when we won a first-price auction), so it is our best
      single-number estimate of the clearing price for that opportunity.
    - ``won`` / ``clicked`` as clean nullable-boolean views.
    """
    out = df.copy()

    out["viewability_missing"] = out["viewability"].eq(SENTINELS["viewability"])
    out["session_depth_missing"] = out["session_depth"].eq(SENTINELS["session_depth"])
    # Replace sentinels with NaN in *analysis* copies so summary stats aren't skewed.
    out["viewability_est"] = out["viewability"].where(~out["viewability_missing"])
    out["session_depth_est"] = out["session_depth"].where(~out["session_depth_missing"])

    out["won"] = out["won_bid"].astype("Float64").astype("boolean")
    out["clicked"] = out["conversion"].astype("Float64").astype("boolean")

    # The price to beat. Only meaningful where we bid and observed feedback.
    out["clearing_price"] = out["feedback_bid"].where(out["bidded"].eq(1))

    return out


def bidded_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Rows where we participated in the auction (bid, feedback observed)."""
    return df[df["bidded"].eq(1)].copy()


def won_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Rows where the ad was actually served, the only rows where CTR is defined."""
    return df[df["won_bid"].eq(1)].copy()
