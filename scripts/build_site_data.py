"""Compute the aggregates the report website charts need and write them to
docs/assets/site_data.json.

The published site renders these with Plotly client-side, so only aggregated
numbers leave the machine, never raw rows. Run after the data is in place:

    python3 scripts/build_site_data.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import bid_landscape as bl  # noqa: E402
import ctr_features as cf  # noqa: E402
import data_loading as dl  # noqa: E402
import layers as ly  # noqa: E402
import profiling as pf  # noqa: E402

OUT = ROOT / "docs" / "assets" / "site_data.json"


def r(x, n=4):
    """Round scalars and leave everything else alone; JSON-friendly.

    NaN and infinities become None so the payload is valid JSON that a browser
    can parse (Python's json writes a bare ``NaN`` otherwise, which JSON.parse
    rejects).
    """
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return None if (np.isnan(x) or np.isinf(x)) else round(x, n)
    if isinstance(x, (int, np.integer)):
        return int(x)
    return x


def main():
    df = dl.add_derived(dl.load_raw())
    bidded = dl.bidded_frame(df)
    won = dl.won_frame(df)
    overall_ctr = float(won["conversion"].mean())
    out = {"meta": {"rows": int(len(df)), "overall_ctr": r(overall_ctr)}}

    # ---- Q1 funnel ----
    fn = pf.funnel(df)
    out["funnel"] = {
        "stages": fn["stage"].tolist(),
        "counts": [int(c) for c in fn["count"]],
        "pct_req": [r(p) for p in fn["pct_of_requests"]],
    }

    # ---- Q1 auction type ----
    at = bidded.groupby("auction_type").agg(
        n=("won_bid", "size"),
        win_rate=("won_bid", "mean"),
        median_bid=("bid", "median"),
        median_feedback=("feedback_bid", "median"),
    )
    at["share"] = at["n"] / len(bidded)
    out["auction"] = [
        {"type": idx, "n": int(row.n), "share": r(row.share), "win_rate": r(row.win_rate),
         "median_bid": r(row.median_bid, 2), "median_feedback": r(row.median_feedback, 2)}
        for idx, row in at.iterrows()
    ]

    # ---- Q1 price distributions (log10 histogram, per auction type) ----
    price = {}
    for col in ["bid", "feedback_bid"]:
        d = bidded[col].dropna()
        d = d[d > 0]
        counts, edges = np.histogram(np.log10(d), bins=50)
        price[col] = {"edges": [r(e, 3) for e in edges], "counts": [int(c) for c in counts],
                      "median": r(float(d.median()), 2)}
    out["price"] = price

    # ---- Q2 information value and mutual information ----
    cat_candidates = ["device_brand", "device_type", "detected_language", "is_wifi",
                      "auction_type", "state_code", "city", "hour",
                      "viewability_missing", "session_depth_missing"]
    num_candidates = ["viewability_est", "session_depth_est", "cookie_age_seconds"]
    iv = cf.rank_categorical_iv(won, cat_candidates)
    out["iv"] = [{"feature": f, "iv": r(v, 4), "strength": str(s)}
                 for f, v, s in zip(iv["feature"], iv["information_value"], iv["strength"])]
    mi = cf.rank_numeric_mi(won, num_candidates)
    out["mi"] = [{"feature": f, "mi": r(v, 5)} for f, v in zip(mi["feature"], mi["mutual_info"])]

    # ---- Q2 CTR trends (session depth, viewability) ----
    w = won.copy()
    w["sd_bin"] = pd.cut(w["session_depth_est"], [0, 1, 2, 3, 5, 10, 1e9],
                         labels=["1", "2", "3", "4-5", "6-10", "10+"])
    sd = w.groupby("sd_bin", observed=True)["conversion"].agg(["mean", "size"]).dropna()
    out["ctr_session"] = [{"bin": str(i), "ctr": r(row["mean"]), "n": int(row["size"])}
                          for i, row in sd.iterrows()]
    w["vb_bin"] = pd.cut(w["viewability_est"], [-1, 25, 50, 75, 90, 100],
                         labels=["0-25", "25-50", "50-75", "75-90", "90-100"])
    vb = w.groupby("vb_bin", observed=True)["conversion"].agg(["mean", "size"]).dropna()
    out["ctr_view"] = [{"bin": str(i), "ctr": r(row["mean"]), "n": int(row["size"])}
                       for i, row in vb.iterrows()]

    # ---- Q3 bid curves and calibration ----
    model = bl.HierarchicalQuantileModel(levels=["url", "ad_slot", "domain"], pseudo_count=30).fit(bidded)
    fp = bidded[(bidded.auction_type == "FIRST_PRICE") & bidded.clearing_price.notna()]
    top_pairs = fp.groupby(["url", "ad_slot"]).size().sort_values(ascending=False).head(4).index
    targets = np.round(np.linspace(0.05, 0.95, 19), 3)
    curves = []
    for i, (u, s) in enumerate(top_pairs):
        row = fp[(fp.url == u) & (fp.ad_slot == s)].iloc[0]
        n = int(((fp.url == u) & (fp.ad_slot == s)).sum())
        curves.append({"label": f"pair {i + 1}", "n": n,
                       "bids": [r(float(model.bid_for_winrate(row, x)), 3) for x in targets]})
    out["bid_curves"] = {"targets": [r(t, 3) for t in targets], "pairs": curves}

    calib = []
    for x in [0.2, 0.35, 0.5, 0.65, 0.8]:
        cal = bl.calibration_by_pair(bidded, model, x, min_n=50)
        calib.append({"target": r(x, 2), "median_realised": r(float(cal["realised_win_rate"].median())),
                      "n_pairs": int(len(cal))})
    out["calibration"] = calib

    # ---- Layers: hourly ----
    hp = ly.hourly_profile(df, bidded, won)
    out["hourly"] = [{"hour": int(row.hour), "requests": int(row.requests),
                      "bid_rate": r(row.bid_rate), "win_rate": r(row.win_rate),
                      "median_cpm": r(row.median_cpm, 3), "ctr": r(row.ctr)}
                     for row in hp.itertuples()]

    # ---- Layers: Lorenz ----
    lorenz = {}
    for col in ["domain", "url", "ad_slot"]:
        counts = df[col].value_counts().to_numpy()
        pts = ly.lorenz_points(counts)
        g = ly.gini(counts)
        # subsample the curve to keep the payload small
        x = pts["share_units"].to_numpy()
        y = pts["share_impressions"].to_numpy()
        idx = np.linspace(0, len(x) - 1, 200).astype(int)
        lorenz[col] = {"x": [r(v, 4) for v in x[idx]], "y": [r(v, 4) for v in y[idx]],
                       "gini": r(g), "n": int(df[col].nunique())}
    out["lorenz"] = lorenz

    # ---- Layers: browser and OS ----
    browser, os_table = ly.browser_os_summary(df, won)
    out["browser"] = [{"name": row.browser, "share": r(row.share), "ctr": r(row.ctr),
                       "served": int(row.served)} for row in browser.itertuples()]
    out["os"] = [{"name": row.os, "share": r(row.share), "ctr": r(row.ctr),
                  "served": int(row.served)} for row in os_table.itertuples()]

    # ---- Layers: cookie age ----
    ck = ly.cookie_age_ctr(won)
    out["cookie_age"] = [{"bin": row.cookie_age, "ctr": r(row.ctr), "served": int(row.served)}
                         for row in ck.itertuples()]

    # ---- Layers: archetypes (per-domain points) ----
    fp_dom = ly.domain_fingerprints(df, bidded, won, min_n=100)
    arch = ly.publisher_archetypes(fp_dom)
    lab = arch["labelled"].dropna(subset=["median_bid", "ctr"])
    lab = lab[lab["median_bid"] > 0]
    out["archetypes"] = {
        "names": {int(k): v for k, v in arch["names"].items()},
        "points": [{"c": int(row.cluster), "mobile": r(row.mobile_share), "bid": r(row.median_bid, 2),
                    "n": int(row.n), "ctr": r(row.ctr)} for row in lab.itertuples()],
    }

    # ---- Layers: state CTR (choropleth) ----
    states = ly.state_summary(df, bidded, won)
    out["states"] = [{"code": row.state_code, "ctr": r(row.ctr), "served": int(row.served),
                      "impressions": int(row.impressions), "win_rate": r(row.win_rate)}
                     for row in states.itertuples() if isinstance(row.state_code, str) and len(row.state_code) == 2]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False turns any stray NaN into an error rather than emitting
    # invalid JSON; r() has already mapped the expected ones to None.
    OUT.write_text(json.dumps(out, separators=(",", ":"), allow_nan=False))
    print(f"Wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB) with keys: {', '.join(out)}")


if __name__ == "__main__":
    main()
