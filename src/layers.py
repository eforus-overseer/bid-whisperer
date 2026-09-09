"""Extra analysis layers beyond the three assignment questions.

Every function here returns a tidy DataFrame or a plain dict so the notebook and
the report generator can display or plot it. Plotting lives in ``layer_plots``.

A note on users
---------------
``user_id`` is unique on every row of this sample (249,232 rows, 249,232 users).
The file is one impression per user, so frequency capping, reach and ad-fatigue
analyses are impossible here. ``user_uniqueness_report`` makes that check explicit
so nobody builds a per-user feature by accident.

Layers
------
1. Bid shading: how much of each winning first-price bid was above the runner-up,
   and what a flat shading policy would have saved.
2. Hourly profile of requests, bid rate, win rate, bid level and CTR (UTC).
3. Traffic concentration across domains, pages and ad slots (Lorenz, Gini).
4. Browser and OS families parsed from the user agent string.
5. CTR by cookie age.
6. Behavioural fingerprints per domain and KMeans publisher archetypes.
7. Per-state summary for the tile map.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FIRST = "FIRST_PRICE"
SECOND = "SECOND_PRICE"


# --------------------------------------------------------------------------- users
def user_uniqueness_report(df: pd.DataFrame) -> dict:
    """Report whether the sample holds one impression per user (it does)."""
    n_rows = int(len(df))
    n_users = int(df["user_id"].nunique())
    return {
        "n_rows": n_rows,
        "n_unique_users": n_users,
        "one_impression_per_user": bool(n_rows == n_users),
        "max_impressions_per_user": int(df["user_id"].value_counts().max()),
    }


# --------------------------------------------------------------------------- 1. bid shading
def overpay_frame(bidded: pd.DataFrame) -> pd.DataFrame:
    """Won first-price rows with the amount paid above the runner-up bid.

    In a first-price auction the winner pays their own bid, and ``feedback_bid``
    is the second-highest bid. Everything between the two is money that did not
    change the outcome.
    """
    mask = bidded["auction_type"].eq(FIRST) & bidded["won_bid"].eq(1)
    fp = bidded.loc[mask].dropna(subset=["bid", "feedback_bid"]).copy()
    fp["overpay"] = fp["bid"] - fp["feedback_bid"]
    fp["overpay_ratio"] = fp["overpay"] / fp["bid"]
    return fp


def lost_gap_frame(bidded: pd.DataFrame) -> pd.DataFrame:
    """Lost first-price rows with the gap between our bid and the winning bid."""
    mask = bidded["auction_type"].eq(FIRST) & bidded["won_bid"].eq(0)
    lost = bidded.loc[mask].dropna(subset=["bid", "feedback_bid"]).copy()
    lost["gap"] = lost["feedback_bid"] - lost["bid"]
    lost["gap_ratio"] = lost["gap"] / lost["bid"]
    return lost


def overpay_stats(bidded: pd.DataFrame) -> dict:
    """Headline numbers for the bid-shading layer."""
    fp = overpay_frame(bidded)
    lost = lost_gap_frame(bidded)
    spend = float(fp["bid"].sum())
    floor = float(fp["feedback_bid"].sum())
    return {
        "n_won_first_price": int(len(fp)),
        "n_won_below_runner_up_anomalies": int((fp["overpay"] < 0).sum()),
        "overpay_cpm_median": float(fp["overpay"].median()),
        "overpay_cpm_mean": float(fp["overpay"].mean()),
        "overpay_ratio_median": float(fp["overpay_ratio"].median()),
        "overpay_ratio_p25": float(fp["overpay_ratio"].quantile(0.25)),
        "overpay_ratio_p75": float(fp["overpay_ratio"].quantile(0.75)),
        "share_paid_over_2x_runner_up": float((fp["bid"] > 2 * fp["feedback_bid"]).mean()),
        "first_price_spend_cpm_sum": spend,
        "runner_up_cpm_sum": floor,
        "theoretical_max_saving_share": float((spend - floor) / spend) if spend else np.nan,
        "n_lost_first_price": int(len(lost)),
        "lost_gap_cpm_median": float(lost["gap"].median()),
        "share_lost_by_under_10pct": float((lost["gap_ratio"] < 0.10).mean()),
        "share_lost_by_under_25pct": float((lost["gap_ratio"] < 0.25).mean()),
    }


def shading_curve(bidded: pd.DataFrame, shade_grid=None) -> pd.DataFrame:
    """What a flat bid-shading policy would have done to wins and spend.

    For each shading level ``s`` we replay every won first-price auction with the
    bid cut to ``bid * (1 - s)``. We keep the win when the cut bid still beats the
    runner-up. ``spend_saved`` compares the new spend on the auctions we keep with
    the original spend on all of them, so it includes the money we stop spending
    on impressions we lose.
    """
    fp = overpay_frame(bidded)
    grid = np.round(np.arange(0.0, 0.71, 0.05), 2) if shade_grid is None else np.asarray(shade_grid)
    bid = fp["bid"].to_numpy()
    runner_up = fp["feedback_bid"].to_numpy()
    spend = bid.sum()
    n = len(bid)
    rows = []
    for s in grid:
        new_bid = bid * (1 - s)
        keep = new_bid > runner_up
        new_spend = new_bid[keep].sum()
        rows.append(
            {
                "shade": float(s),
                "win_retained": float(keep.mean()),
                "spend_saved": float(1 - new_spend / spend),
                "cost_per_win_before": float(spend / n),
                "cost_per_win_after": float(new_spend / max(int(keep.sum()), 1)),
                "impressions_lost": int((~keep).sum()),
            }
        )
    return pd.DataFrame(rows)


def oracle_curve(bidded: pd.DataFrame, eps_grid=None) -> pd.DataFrame:
    """Ceiling on savings: bid the runner-up price plus ``eps`` in every won auction.

    This needs knowledge nobody has before the auction, so it is a bound, not a
    policy. Every auction is still won by construction.
    """
    fp = overpay_frame(bidded)
    grid = np.round(np.arange(0.0, 0.51, 0.05), 2) if eps_grid is None else np.asarray(eps_grid)
    bid = fp["bid"].to_numpy()
    runner_up = fp["feedback_bid"].to_numpy()
    spend = bid.sum()
    rows = [
        {"eps": float(e), "spend_saved": float(1 - np.minimum(bid, runner_up * (1 + e)).sum() / spend)}
        for e in grid
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 2. hourly
def hourly_profile(df: pd.DataFrame, bidded: pd.DataFrame, won: pd.DataFrame) -> pd.DataFrame:
    """Requests, bid rate, win rate, median bid and CTR for each UTC hour."""
    h = df.groupby("hour").agg(requests=("bidded", "size"), bid_rate=("bidded", "mean"))
    hb = bidded.groupby("hour").agg(win_rate=("won_bid", "mean"), median_cpm=("bid", "median"))
    hw = won.groupby("hour").agg(ctr=("conversion", "mean"), served=("conversion", "size"))
    return h.join(hb).join(hw).reset_index()


# --------------------------------------------------------------------------- 3. concentration
def gini(counts) -> float:
    """Gini coefficient of a vector of counts (0 = equal, 1 = one unit has all)."""
    x = np.sort(np.asarray(counts, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return float("nan")
    return float((2 * np.sum(np.arange(1, n + 1) * x) / (n * x.sum())) - (n + 1) / n)


def lorenz_points(counts) -> pd.DataFrame:
    """Lorenz curve points: units ranked smallest to largest vs cumulative volume."""
    x = np.sort(np.asarray(counts, dtype=float))
    cum = np.cumsum(x) / x.sum()
    units = np.arange(1, len(x) + 1) / len(x)
    return pd.DataFrame({"share_units": np.r_[0.0, units], "share_impressions": np.r_[0.0, cum]})


def concentration_table(df: pd.DataFrame, cols=("domain", "url", "ad_slot"), ks=(10, 50, 100, 500)) -> pd.DataFrame:
    """Top-k volume shares and Gini for each identifier column."""
    rows = []
    for c in cols:
        vc = df[c].value_counts()
        cum = vc.cumsum() / vc.sum()
        row = {"column": c, "n_unique": int(len(vc)), "gini": gini(vc.to_numpy())}
        for k in ks:
            row[f"top_{k}_share"] = float(cum.iloc[min(k, len(vc)) - 1])
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 4. browser / OS
# Order matters: Edge and Opera also contain "Chrome"; Chrome also contains "Safari".
_BROWSER_RULES = [
    ("Edge", r"Edg[eA]?/"),
    ("Opera", r"OPR/|Opera"),
    ("Firefox", r"Firefox/|FxiOS/"),
    ("Chrome", r"Chrome/|CriOS/"),
    ("Safari", r"Safari/"),
]
# iPads on iPadOS 13+ present themselves as Macintosh, so a little iOS traffic lands in macOS.
_OS_RULES = [
    ("Windows", r"Windows"),
    ("iOS", r"iPhone|iPad|iPod|CriOS/|FxiOS/"),
    ("Android", r"Android"),
    ("macOS", r"Macintosh|Mac OS X"),
    ("ChromeOS", r"CrOS"),
    ("Linux", r"Linux"),
]


def _classify(s: pd.Series, rules) -> pd.Series:
    out = pd.Series("Other", index=s.index, dtype="object")
    assigned = pd.Series(False, index=s.index)
    for name, pattern in rules:
        m = s.str.contains(pattern, regex=True) & ~assigned
        out[m] = name
        assigned |= m
    return out


def parse_user_agent(ua: pd.Series) -> pd.DataFrame:
    """Browser and OS family for each user agent string."""
    s = ua.fillna("").astype(str)
    return pd.DataFrame({"browser": _classify(s, _BROWSER_RULES), "os": _classify(s, _OS_RULES)}, index=ua.index)


def browser_os_summary(df: pd.DataFrame, won: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Share of impressions and CTR (on served impressions) per browser and OS."""
    parsed = parse_user_agent(df["user_agent"])
    out = []
    for col in ("browser", "os"):
        n = parsed[col].value_counts().rename("n")
        share = parsed[col].value_counts(normalize=True).rename("share")
        w = won[["conversion"]].join(parsed[col])
        ctr = w.groupby(col)["conversion"].agg(served="size", ctr="mean")
        table = pd.concat([n, share, ctr], axis=1).rename_axis(col).reset_index()
        out.append(table.sort_values("share", ascending=False).reset_index(drop=True))
    return out[0], out[1]


# --------------------------------------------------------------------------- 5. cookie age
COOKIE_BINS = [-np.inf, 0, 1, 7, 30, 90, 365, np.inf]
COOKIE_LABELS = ["new (0)", "under 1 day", "1 to 7 days", "7 to 30 days", "30 to 90 days", "90 to 365 days", "over 1 year"]


def cookie_age_ctr(won: pd.DataFrame) -> pd.DataFrame:
    """CTR on served impressions by cookie-age bucket (days since the id was created)."""
    days = won["cookie_age_seconds"] / 86400
    bucket = pd.cut(days, COOKIE_BINS, labels=COOKIE_LABELS)
    g = won.groupby(bucket, observed=False)["conversion"].agg(served="size", clicks="sum", ctr="mean")
    return g.rename_axis("cookie_age").reset_index()


# --------------------------------------------------------------------------- 6. archetypes
FINGERPRINT_FEATURES = [
    "mobile_share",
    "non_en_share",
    "viewability_mean",
    "session_depth_median",
    "second_price_share",
    "wifi_share",
    "median_bid",
    "win_rate",
    "ctr",
    "peak_sin",
    "peak_cos",
]

# Phrase for a high (positive z) and a low (negative z) centroid value of each feature.
_PHRASES = {
    "mobile_share": ("mobile", "desktop"),
    "median_bid": ("premium CPM", "cheap inventory"),
    "second_price_share": ("second-price", None),
    "viewability_mean": ("high viewability", "low viewability"),
    "session_depth_median": ("deep sessions", "shallow sessions"),
    "ctr": ("high CTR", "low CTR"),
    "win_rate": ("easy to win", "contested"),
    "non_en_share": ("non-English", None),
    "wifi_share": (None, "more cellular"),
}


def domain_fingerprints(df: pd.DataFrame, bidded: pd.DataFrame, won: pd.DataFrame, min_n: int = 100) -> pd.DataFrame:
    """One behavioural row per domain with at least ``min_n`` impressions.

    The hashes hide who the publisher is, but how their traffic behaves is in the
    clear: device mix, language, viewability, session depth, auction type,
    connection, price level, how often we win, how often users click, and the
    hour their traffic peaks (encoded on a circle so 23h sits next to 0h).
    """
    d = df.assign(
        is_mobile=df["device_type"].ne("PERSONAL_COMPUTER"),
        non_en=df["detected_language"].ne("en") & df["detected_language"].notna(),
        is_second=df["auction_type"].eq(SECOND),
    )
    g = d.groupby("domain").agg(
        n=("bidded", "size"),
        bid_rate=("bidded", "mean"),
        mobile_share=("is_mobile", "mean"),
        non_en_share=("non_en", "mean"),
        viewability_mean=("viewability_est", "mean"),
        session_depth_median=("session_depth_est", "median"),
        second_price_share=("is_second", "mean"),
        wifi_share=("is_wifi", "mean"),
    )
    gb = bidded.groupby("domain").agg(n_bidded=("bid", "size"), median_bid=("bid", "median"), win_rate=("won_bid", "mean"))
    gw = won.groupby("domain").agg(n_won=("conversion", "size"), ctr=("conversion", "mean"))
    peak = (
        d.groupby(["domain", "hour"]).size().rename("c").reset_index()
        .sort_values(["domain", "c"], ascending=[True, False])
        .drop_duplicates("domain")
        .set_index("domain")["hour"]
        .rename("peak_hour")
    )
    fp = g.join(gb).join(gw).join(peak)
    fp = fp[fp["n"] >= min_n].copy()
    angle = 2 * np.pi * fp["peak_hour"] / 24
    fp["peak_sin"] = np.sin(angle)
    fp["peak_cos"] = np.cos(angle)
    return fp


def _name_cluster(z: pd.Series, n_terms: int = 2) -> str:
    """Describe a centroid by its two most unusual features (in z-score units)."""
    order = z.drop(labels=["peak_sin", "peak_cos"]).abs().sort_values(ascending=False).index
    terms: list[str] = []
    for f in order:
        hi, lo = _PHRASES.get(f, (None, None))
        phrase = hi if z[f] > 0 else lo
        if phrase and phrase not in terms:
            terms.append(phrase)
        if len(terms) == n_terms:
            break
    return ", ".join(terms) if terms else "average"


def publisher_archetypes(fp: pd.DataFrame, k_range=range(3, 7), seed: int = 0) -> dict:
    """KMeans on standardised fingerprints; k picked by silhouette.

    Returns the labelled frame, centroids in original units and in z units, the
    silhouette table, the chosen k and a name per cluster.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    X = fp[FINGERPRINT_FEATURES].copy()
    X = X.fillna(X.median(numeric_only=True))
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)

    sil, best = [], None
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
        score = float(silhouette_score(Z, km.labels_))
        sil.append({"k": k, "silhouette": score})
        if best is None or score > best[0]:
            best = (score, k, km)
    _, k, km = best

    labelled = fp.copy()
    labelled["cluster"] = km.labels_
    centers_z = pd.DataFrame(km.cluster_centers_, columns=FINGERPRINT_FEATURES)
    centers = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=FINGERPRINT_FEATURES)

    names = {i: _name_cluster(centers_z.iloc[i]) for i in range(k)}
    # Make names unique by adding a third descriptor where two clusters collide.
    seen = pd.Series(names).value_counts()
    for i in range(k):
        if seen[names[i]] > 1:
            names[i] = _name_cluster(centers_z.iloc[i], n_terms=3)

    labelled["archetype"] = labelled["cluster"].map(names)
    centers["archetype"] = [names[i] for i in range(k)]
    centers["n_domains"] = labelled["cluster"].value_counts().sort_index().to_numpy()
    centers["impressions"] = labelled.groupby("cluster")["n"].sum().sort_index().to_numpy()
    return {
        "labelled": labelled,
        "centroids": centers,
        "centroids_z": centers_z,
        "silhouette": pd.DataFrame(sil),
        "best_k": int(k),
        "names": names,
    }


# --------------------------------------------------------------------------- 7. states
def state_summary(df: pd.DataFrame, bidded: pd.DataFrame, won: pd.DataFrame) -> pd.DataFrame:
    """Impressions, bid rate, win rate, served impressions, clicks and CTR per state."""
    a = df.groupby("state_code").agg(impressions=("bidded", "size"), bid_rate=("bidded", "mean"))
    b = bidded.groupby("state_code").agg(bidded=("won_bid", "size"), win_rate=("won_bid", "mean"), median_bid=("bid", "median"))
    w = won.groupby("state_code").agg(served=("conversion", "size"), clicks=("conversion", "sum"), ctr=("conversion", "mean"))
    out = a.join(b).join(w).reset_index()
    return out.sort_values("impressions", ascending=False).reset_index(drop=True)
