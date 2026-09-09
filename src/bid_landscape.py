"""Pre-bid win-rate / bid-landscape estimation (Q3).

Problem restated
----------------
For every (url, ad_slot) we want a function ``bid(X)`` -> the CPM we should bid to
win the auction X% of the time, computable *before* we bid.

Insight: to win, our bid must exceed the market **clearing price** for that slot.
We reconstruct the clearing price from ``feedback_bid`` (see data_loading). Then

    P(win | bid) = P(bid > clearing_price) = CDF of clearing price
    => bid to win X% of the time = the X-th percentile of the clearing-price
       distribution for that (url, ad_slot).

So Q3 reduces to **quantile estimation of clearing price per (url, ad_slot)**.

The obstacle is sparsity: the median (url, ad_slot) pair has ~2 observations, so a
raw empirical quantile is hopelessly noisy. We solve this with **hierarchical
partial pooling / empirical-Bayes shrinkage**: blend the pair-level quantile toward
progressively broader fallbacks (url -> ad_slot -> domain -> auction_type -> global),
weighting each level by its support. This is the same idea as shrinking a noisy
per-item rate toward a population prior.

Auction type is handled first-class: FIRST_PRICE and SECOND_PRICE have wildly
different clearing-price regimes, so pooling never crosses that boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class HierarchicalQuantileModel:
    """Empirical-Bayes-style hierarchical clearing-price quantile estimator.

    Parameters
    ----------
    levels : list[str]
        Grouping keys from most specific to most general. Each successively coarser
        level acts as the prior for the finer one.
    pseudo_count : float
        Strength of the prior. A level's estimate is credible once it has roughly
        this many observations; below that it is shrunk toward its parent.
    """

    levels: list[str] = field(default_factory=lambda: ["url", "ad_slot", "domain"])
    price_col: str = "clearing_price"
    strata_col: str = "auction_type"
    pseudo_count: float = 30.0
    _tables: dict = field(default_factory=dict, init=False)
    _global: dict = field(default_factory=dict, init=False)

    def fit(self, df: pd.DataFrame) -> "HierarchicalQuantileModel":
        data = df[df[self.price_col].notna()].copy()
        # Global fallback quantile function, per auction stratum.
        self._global = {
            stratum: g[self.price_col].to_numpy()
            for stratum, g in data.groupby(self.strata_col)
        }
        self._all_prices = data[self.price_col].to_numpy()
        # Store raw price arrays keyed by (stratum, level, level_value).
        self._tables = {}
        for lvl in self.levels:
            tbl = {}
            for (stratum, val), g in data.groupby([self.strata_col, lvl]):
                tbl[(stratum, val)] = g[self.price_col].to_numpy()
            self._tables[lvl] = tbl
        self._data = data
        return self

    @staticmethod
    def _q(arr: np.ndarray, x: float) -> float:
        return float(np.quantile(arr, x)) if len(arr) else np.nan

    def bid_for_winrate(self, row: pd.Series, target_winrate: float) -> float:
        """Shrunk estimate of the bid needed to win ``target_winrate`` of the time.

        Walks specific -> general, accumulating a support-weighted blend of the
        per-level empirical quantiles. Levels with more data dominate.
        """
        stratum = row[self.strata_col]
        num, den = 0.0, 0.0
        for lvl in self.levels:
            arr = self._tables.get(lvl, {}).get((stratum, row[lvl]))
            if arr is not None and len(arr):
                w = len(arr) / (len(arr) + self.pseudo_count)
                q = self._q(arr, target_winrate)
                # Remaining mass flows to coarser levels.
                num += (1 - den) * w * q
                den += (1 - den) * w
        # Global stratum fallback for any residual mass.
        g_arr = self._global.get(stratum, self._all_prices)
        num += (1 - den) * self._q(g_arr, target_winrate)
        return num

    def predict_frame(self, keys: pd.DataFrame, target_winrate: float) -> pd.Series:
        return keys.apply(lambda r: self.bid_for_winrate(r, target_winrate), axis=1)


def empirical_winrate_curve(prices: np.ndarray, bids: np.ndarray) -> pd.DataFrame:
    """Realised win rate as a function of bid threshold, for validation plots.

    Given observed clearing ``prices``, the win rate achievable by bidding ``b`` is
    the fraction of prices strictly below ``b`` (P(b > clearing_price)).
    """
    prices = np.sort(prices)
    wr = np.searchsorted(prices, bids, side="left") / len(prices)
    return pd.DataFrame({"bid": bids, "win_rate": wr})


def calibration_by_pair(
    df: pd.DataFrame,
    model: HierarchicalQuantileModel,
    target_winrate: float,
    group_keys: list[str] | None = None,
    min_n: int = 30,
) -> pd.DataFrame:
    """Back-test: for well-populated pairs, does bidding the model's price actually
    win ~``target_winrate`` of the time? Returns realised vs. target win rate.
    """
    group_keys = group_keys or ["url", "ad_slot"]
    data = df[df["clearing_price"].notna()]
    rows = []
    for keys, g in data.groupby(group_keys):
        if len(g) < min_n:
            continue
        rep = g.iloc[0]
        pred_bid = model.bid_for_winrate(rep, target_winrate)
        realised = float((pred_bid > g["clearing_price"]).mean())
        rows.append((*(keys if isinstance(keys, tuple) else (keys,)), len(g), pred_bid, realised))
    cols = group_keys + ["n", "predicted_bid", "realised_win_rate"]
    return pd.DataFrame(rows, columns=cols)
