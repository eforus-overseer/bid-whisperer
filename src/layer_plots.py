"""Figures for the extra analysis layers and the obfuscation forensics.

Every function returns a matplotlib Figure and draws nothing else. Chart forms
follow the Visual Vocabulary families (distribution, change over time,
part-to-whole, magnitude, correlation, spatial). House rules: fixed categorical
order from ``viz_style.CATEGORICAL``, one y axis per panel, a single blue hue for
magnitude ramps, direct labels where they help, sentence-case titles.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

import viz_style as vs

# Single-hue blue ramp, light to dark (same steps as the validated dataviz palette).
SEQ_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
SEQ_CMAP = mpl.colors.LinearSegmentedColormap.from_list("nsrb_seq", SEQ_RAMP)
SHADE = "#efeee9"  # quiet background band
SOURCE = "Source: one day of RTB bid-request logs (2021-02-01, US only)."
PCT0 = mpl.ticker.PercentFormatter(1.0, decimals=0)
PCT1 = mpl.ticker.PercentFormatter(1.0, decimals=1)


def _source(fig, text: str = SOURCE, y: float = -0.01) -> None:
    """Source line under the figure; lower ``y`` when tall tick labels need the room."""
    fig.text(0.01, y, text, fontsize=7.5, color=vs.INK_MUTED, va="top", ha="left")


def _ink_for(face) -> str:
    """White text on dark fills, ink on light ones."""
    r, g, b = mpl.colors.to_rgb(face)
    return "#ffffff" if (0.2126 * r + 0.7152 * g + 0.0722 * b) < 0.5 else vs.INK


# --------------------------------------------------------------------------- distribution
def plot_overpay(fp: pd.DataFrame, curve: pd.DataFrame, stats: dict) -> plt.Figure:
    """Left: distribution of overpay. Right: the flat bid-shading replay."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ratio = fp["overpay_ratio"].clip(lower=0, upper=1)
    ax.hist(ratio, bins=40, color=vs.SEQUENTIAL)
    med = stats["overpay_ratio_median"]
    ax.axvline(med, color=vs.INK, ls="--", lw=1)
    ax.text(med + 0.015, ax.get_ylim()[1] * 0.92, f"median: {med:.0%} of the bid\nwas above the runner-up",
            fontsize=8.5, color=vs.INK_SECONDARY, va="top")
    ax.xaxis.set_major_formatter(PCT0)
    ax.grid(False)
    vs.despine_labels(ax, "Overpay in won first-price auctions",
                      "overpay as a share of our bid", "won first-price auctions")

    ax = axes[1]
    ax.plot(curve["shade"], curve["win_retained"], color=vs.CATEGORICAL[0], lw=2, marker="o", ms=3.5)
    ax.plot(curve["shade"], curve["spend_saved"], color=vs.CATEGORICAL[1], lw=2, marker="o", ms=3.5)
    last = curve.iloc[-1]
    ax.text(last["shade"] + 0.01, last["win_retained"], "wins kept", color=vs.CATEGORICAL[0], fontsize=9, va="center")
    ax.text(last["shade"] + 0.01, last["spend_saved"], "spend saved", color=vs.CATEGORICAL[1], fontsize=9, va="center")
    ten = curve.loc[np.isclose(curve["shade"], 0.10)]
    if len(ten):
        r = ten.iloc[0]
        ax.axvline(0.10, color=vs.INK_MUTED, lw=0.8, ls=":")
        # Callout lives in the empty top-middle of the panel, above both curves.
        ax.annotate(f"shade every bid by 10%:\nkeep {r.win_retained:.0%} of wins, save {r.spend_saved:.0%} of spend",
                    (0.10, r.win_retained), xytext=(0.24, 0.96), textcoords="data", fontsize=8.5,
                    color=vs.INK_SECONDARY, va="top",
                    arrowprops=dict(arrowstyle="-", color=vs.INK_MUTED, lw=0.8, shrinkB=3))
    ax.set_xlim(0, curve["shade"].max() + 0.14)
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_formatter(PCT0)
    ax.yaxis.set_major_formatter(PCT0)
    vs.despine_labels(ax, "Replay: every bid cut by a flat percentage",
                      "uniform bid shading", "share of the original")
    _source(fig)
    return fig


# --------------------------------------------------------------------------- change over time
def plot_hourly(hp: pd.DataFrame) -> plt.Figure:
    """Five small multiples on a shared hour axis."""
    panels = [
        ("requests", "Requests", "{:,.0f}", None),
        ("bid_rate", "Bid rate (share of requests we bid on)", "{:.0%}", PCT0),
        ("win_rate", "Win rate (share of bids won)", "{:.0%}", PCT0),
        ("median_cpm", "Median bid (CPM)", "{:.2f}", None),
        ("ctr", "CTR on served impressions", "{:.1%}", PCT1),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 13), sharex=True)
    for ax, (col, title, fmt, formatter) in zip(axes, panels):
        ax.axvspan(7.5, 10.5, color=SHADE, zorder=0, lw=0)
        ax.plot(hp["hour"], hp[col], color=vs.SEQUENTIAL, lw=2, marker="o", ms=3.5)
        imax, imin = hp[col].idxmax(), hp[col].idxmin()
        ax.annotate(fmt.format(hp[col][imax]), (hp["hour"][imax], hp[col][imax]), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8, color=vs.INK_SECONDARY)
        ax.annotate(fmt.format(hp[col][imin]), (hp["hour"][imin], hp[col][imin]), textcoords="offset points",
                    xytext=(0, -13), ha="center", fontsize=8, color=vs.INK_SECONDARY)
        ax.margins(y=0.3)
        if formatter is not None:
            ax.yaxis.set_major_formatter(formatter)
        ax.set_title(title, loc="left", fontsize=10, pad=4)
    axes[0].text(9, axes[0].get_ylim()[1] * 0.97, "US night\n(08 to 10 UTC)", ha="center", va="top",
                 fontsize=8, color=vs.INK_MUTED)
    axes[-1].set_xticks(range(0, 24, 2))
    axes[-1].set_xlabel("hour of day (UTC)")
    fig.suptitle("One day in the auction, hour by hour", x=0.01, ha="left", weight="bold", fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0.01, 1, 0.98))
    _source(fig)
    return fig


def plot_cookie_age(ck: pd.DataFrame, overall_ctr: float) -> plt.Figure:
    """CTR across ordered cookie-age buckets with support printed under each point."""
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(len(ck))
    ax.plot(x, ck["ctr"], color=vs.SEQUENTIAL, lw=2, marker="o", ms=7)
    ax.axhline(overall_ctr, color=vs.INK_MUTED, ls="--", lw=1)
    ax.text(x[-1] + 0.15, overall_ctr, f"all served\n{overall_ctr:.1%}", fontsize=8.5, color=vs.INK_MUTED, va="center")
    for i, c in enumerate(ck["ctr"]):
        ax.annotate(f"{c:.1%}", (i, c), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9, color=vs.INK)
    # Support goes into the tick label so nothing sits on the line itself.
    labels = [f"{str(a).replace(' to ', ' to\n').replace('under ', 'under\n').replace('over ', 'over\n')}\nn={int(n):,}"
              for a, n in zip(ck["cookie_age"], ck["served"])]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_xlim(-0.5, len(ck) - 0.5 + 1.0)
    ax.margins(y=0.3)
    ax.yaxis.set_major_formatter(PCT0)
    vs.despine_labels(ax, "Click-through rate by cookie age", "time since the cookie was created (n = served impressions)", "CTR")
    _source(fig, y=-0.12)
    return fig


# --------------------------------------------------------------------------- part-to-whole
def plot_lorenz(curves: dict, n_domains: int) -> plt.Figure:
    """Lorenz curves for the identifier columns, equality line dashed."""
    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    ax.plot([0, 1], [0, 1], color=vs.INK_MUTED, ls="--", lw=1)
    ax.text(0.52, 0.47, "equal shares", rotation=45, fontsize=8, color=vs.INK_MUTED, ha="center", va="top")
    for i, (name, (pts, g)) in enumerate(curves.items()):
        ax.plot(pts["share_units"], pts["share_impressions"], color=vs.CATEGORICAL[i], lw=2,
                label=f"{name}  (Gini {g:.2f})")
    x50 = 1 - 50 / n_domains
    ax.annotate("the 50 biggest domains carry\nhalf of all impressions", (x50, 0.5), textcoords="offset points",
                xytext=(-150, -6), fontsize=8.5, color=vs.INK_SECONDARY,
                arrowprops=dict(arrowstyle="-", color=vs.INK_MUTED, lw=0.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.xaxis.set_major_formatter(PCT0)
    ax.yaxis.set_major_formatter(PCT0)
    ax.legend(loc="upper left", title="ranked smallest to largest")
    vs.despine_labels(ax, "A handful of publishers carry most of the volume",
                      "share of domains, pages or slots", "share of impressions")
    _source(fig)
    return fig


# --------------------------------------------------------------------------- magnitude
def plot_browser_os(browser: pd.DataFrame, os_table: pd.DataFrame, overall_ctr: float, min_served: int = 300) -> plt.Figure:
    """Two rows (browser, OS) of two sorted bar panels (share, CTR)."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    for row, (name, table) in enumerate([("browser", browser), ("os", os_table)]):
        d = table.sort_values("share", ascending=True)
        weak = d["served"].fillna(0) < min_served

        ax = axes[row, 0]
        ax.barh(d[name], d["share"], color=vs.SEQUENTIAL)
        for y, v in enumerate(d["share"]):
            ax.text(v + 0.006, y, f"{v:.1%}", va="center", fontsize=8.5, color=vs.INK_SECONDARY)
        ax.set_xlim(0, d["share"].max() * 1.22)
        ax.xaxis.set_major_formatter(PCT0)
        ax.grid(False)
        vs.despine_labels(ax, f"Share of impressions by {name.upper() if name == 'os' else name}", None, None)

        ax = axes[row, 1]
        colors = [vs.INK_MUTED if w else vs.CATEGORICAL[2] for w in weak]
        ax.barh(d[name], d["ctr"].fillna(0), color=colors)
        ax.axvline(overall_ctr, color=vs.INK, lw=0.9, ls="--")
        for y, (v, w, n) in enumerate(zip(d["ctr"].fillna(0), weak, d["served"].fillna(0))):
            note = f"{v:.1%}" + (f"  (n={int(n)}, thin)" if w else "")
            ax.text(v + 0.003, y, note, va="center", fontsize=8.5, color=vs.INK_SECONDARY)
        ax.set_xlim(0, max(d["ctr"].fillna(0).max(), overall_ctr) * 1.45)
        ax.xaxis.set_major_formatter(PCT0)
        ax.grid(False)
        vs.despine_labels(ax, f"CTR on served impressions by {name.upper() if name == 'os' else name}", None, None)
    axes[0, 1].text(overall_ctr, len(browser) - 0.35, f" all served {overall_ctr:.1%}", fontsize=8, color=vs.INK_SECONDARY, va="center")
    fig.tight_layout()
    _source(fig)
    return fig


# --------------------------------------------------------------------------- correlation
def plot_archetypes(labelled: pd.DataFrame, names: dict, n_label: int = 6) -> plt.Figure:
    """Bubble scatter of domains: device mix vs price level, coloured by archetype."""
    fig, ax = plt.subplots(figsize=(11, 7))
    d = labelled.dropna(subset=["median_bid"])
    d = d[d["median_bid"] > 0]
    for c in sorted(d["cluster"].unique()):
        sub = d[d["cluster"] == c]
        ax.scatter(sub["mobile_share"], sub["median_bid"], s=np.sqrt(sub["n"]) * 3.2, color=vs.CATEGORICAL[c],
                   alpha=0.72, edgecolor=vs.SURFACE, linewidth=0.8, label=f"{names[c]}  ({len(sub)} domains)")
    ax.set_yscale("log")
    # Label the biggest domains. Bubbles pile up at 0% and 100% mobile, so labels on
    # each side are staggered vertically in order of price to keep them apart.
    top = d.sort_values("n", ascending=False).head(n_label)
    offsets = (8, -13, 22, -27, 36, -41)
    for side, group in (("left", top[top["mobile_share"] < 0.5]), ("right", top[top["mobile_share"] >= 0.5])):
        group = group.sort_values("median_bid")
        for k, (dom, row) in enumerate(group.iterrows()):
            dx = 9 if side == "left" else -9
            ax.annotate(dom[:6], (row["mobile_share"], row["median_bid"]), textcoords="offset points",
                        xytext=(dx, offsets[k % len(offsets)]), fontsize=7.5, color=vs.INK_SECONDARY,
                        ha="left" if dx > 0 else "right",
                        arrowprops=dict(arrowstyle="-", color=vs.INK_MUTED, lw=0.6, shrinkB=4))
    ax.xaxis.set_major_formatter(PCT0)
    ax.set_xlim(-0.04, 1.04)
    ax.legend(title="Publisher archetype (bubble area = impressions)", loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, -0.16), fontsize=8.5, title_fontsize=9)
    vs.despine_labels(ax, "Publisher archetypes: what the hashed domains look like from the outside",
                      "share of the domain's impressions on phones or tablets", "median bid (CPM, log scale)")
    fig.tight_layout()
    _source(fig)
    return fig


# --------------------------------------------------------------------------- spatial
# Tile grid map of the 50 states and DC: (column, row), row 0 at the top.
STATE_GRID = {
    "AK": (0, 0), "ME": (11, 0),
    "VT": (10, 1), "NH": (11, 1),
    "WA": (0, 2), "ID": (1, 2), "MT": (2, 2), "ND": (3, 2), "MN": (4, 2), "IL": (5, 2), "WI": (6, 2),
    "MI": (7, 2), "NY": (8, 2), "MA": (9, 2), "RI": (10, 2),
    "OR": (0, 3), "NV": (1, 3), "WY": (2, 3), "SD": (3, 3), "IA": (4, 3), "IN": (5, 3), "OH": (6, 3),
    "PA": (7, 3), "NJ": (8, 3), "CT": (9, 3),
    "CA": (0, 4), "UT": (1, 4), "CO": (2, 4), "NE": (3, 4), "MO": (4, 4), "KY": (5, 4), "WV": (6, 4),
    "VA": (7, 4), "MD": (8, 4), "DE": (9, 4),
    "AZ": (1, 5), "NM": (2, 5), "KS": (3, 5), "AR": (4, 5), "TN": (5, 5), "NC": (6, 5), "SC": (7, 5), "DC": (8, 5),
    "OK": (3, 6), "LA": (4, 6), "MS": (5, 6), "AL": (6, 6), "GA": (7, 6),
    "HI": (0, 7), "TX": (3, 7), "FL": (8, 7),
}
assert len(STATE_GRID) == 51


def plot_state_tile_map(states: pd.DataFrame, min_served: int = 300) -> plt.Figure:
    """US tile grid map coloured by CTR; thin states hatched and left unrated."""
    fig, ax = plt.subplots(figsize=(11.5, 8))
    vals = states.set_index("state_code")
    rated = vals[vals["served"].fillna(0) >= min_served]
    norm = mpl.colors.Normalize(vmin=rated["ctr"].min(), vmax=rated["ctr"].max())
    size = 0.94
    for code, (col, row) in STATE_GRID.items():
        x, y = col, -row
        has = code in vals.index
        served = int(vals.loc[code, "served"]) if has and pd.notna(vals.loc[code, "served"]) else 0
        if has and served >= min_served:
            ctr = float(vals.loc[code, "ctr"])
            face = SEQ_CMAP(norm(ctr))
            ax.add_patch(Rectangle((x, y), size, size, facecolor=face, edgecolor=vs.SURFACE, lw=1.5))
            ink = _ink_for(face)
            ax.text(x + size / 2, y + 0.60, code, ha="center", va="center", fontsize=10, weight="bold", color=ink)
            ax.text(x + size / 2, y + 0.30, f"{ctr:.1%}", ha="center", va="center", fontsize=7.5, color=ink)
        else:
            ax.add_patch(Rectangle((x, y), size, size, facecolor="#f1f0eb", edgecolor="#c9c8c1", lw=1.0, hatch="///"))
            ax.text(x + size / 2, y + 0.60, code, ha="center", va="center", fontsize=10, weight="bold", color=vs.INK_MUTED)
            ax.text(x + size / 2, y + 0.30, f"n={served}", ha="center", va="center", fontsize=7, color=vs.INK_MUTED)
    ax.set_xlim(-0.2, 12.2)
    ax.set_ylim(-7.4, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Click-through rate by state", loc="left", pad=6)
    ax.text(-0.2, 1.05, f"Hatched tiles have fewer than {min_served} served impressions, so no rate is shown.",
            fontsize=9, color=vs.INK_SECONDARY, va="bottom")
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=SEQ_CMAP)
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.035, pad=0.01, shrink=0.45, anchor=(0.9, 1.0))
    cb.ax.xaxis.set_major_formatter(PCT1)
    cb.set_label("CTR on served impressions", fontsize=9, color=vs.INK_SECONDARY)
    cb.outline.set_visible(False)
    _source(fig)
    return fig


# --------------------------------------------------------------------------- forensics
def plot_fd_matrix(fd: pd.DataFrame) -> plt.Figure:
    """Heatmap: share of X values (rows) that map to exactly one Y value (columns)."""
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    m = fd.to_numpy(dtype=float)
    im = ax.imshow(m, cmap=SEQ_CMAP, vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(fd.columns)))
    ax.set_xticklabels(fd.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(fd.index)))
    ax.set_yticklabels(fd.index)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if i == j:
                ax.text(j, i, "self", ha="center", va="center", fontsize=8, color="#ffffff")
            else:
                ax.text(j, i, f"{m[i, j]:.3f}", ha="center", va="center", fontsize=10, color=_ink_for(SEQ_CMAP((m[i, j] - 0.5) / 0.5)))
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("share of row values with a single column value", fontsize=8.5, color=vs.INK_SECONDARY)
    cb.outline.set_visible(False)
    vs.despine_labels(ax, None, "this hashed field is fixed", "given this hashed field")
    ax.set_title("Does knowing one hash pin down another?", loc="left", pad=26)
    ax.text(0, 1.02, "1.000 means every value of the row field appears with exactly one value of the column field.",
            transform=ax.transAxes, fontsize=8, color=vs.INK_SECONDARY, va="bottom")
    return fig


def plot_nibbles(nib: pd.DataFrame) -> plt.Figure:
    """First-hex-digit frequencies per hashed column against the uniform 1/16 line."""
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6), sharey=True)
    for ax, col in zip(axes.ravel(), nib.columns):
        ax.bar(nib.index.astype(str), nib[col], color=vs.SEQUENTIAL, width=0.75)
        ax.axhline(1 / 16, color=vs.INK, ls="--", lw=1)
        ax.set_ylim(0, 0.1)
        ax.yaxis.set_major_formatter(PCT1)
        ax.grid(False)
        vs.despine_labels(ax, col, "first hex digit", "share of distinct values")
    axes[0, 0].text(15.4, 0.076, "dashed line: uniform 1/16", fontsize=8, color=vs.INK_SECONDARY, ha="right")
    fig.suptitle("The hashes look like real digests: every first digit is equally common",
                 x=0.01, ha="left", weight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0.01, 1, 0.96))
    _source(fig, "Distinct values per column. A structured or truncated identifier would show peaks; a hash shows a flat line.")
    return fig
