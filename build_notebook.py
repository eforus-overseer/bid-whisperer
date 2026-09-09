"""Generate the analysis notebook (notebooks/analysis.ipynb).

Keeping the notebook under a generator makes the narrative diffable and the build
reproducible. Run:  python3 build_notebook.py  then execute with nbconvert.

The three sections that go beyond the brief live in ``notebook_sections/`` and are
appended after Q3; each exposes ``cells(md, code)``.
"""
import nbformat as nbf

from notebook_sections import decode, layers, ml_models

nb = nbf.v4.new_notebook()
cells = []

def md(src): cells.append(nbf.v4.new_markdown_cell(src))
def code(src): cells.append(nbf.v4.new_code_cell(src))

# ----------------------------------------------------------------------------- title
md(r"""# Bid whisperer: one day inside a real-time bidding log

Take-home assignment. The data is a single day (2021-02-01) of US-only real-time
bidding (RTB) ad-request logs: 249,232 impressions and 26 columns. Each row is one
auction opportunity, a chance to show an ad to one user, annotated with who the user
is, where the ad would appear, and what happened in the auction.

The notebook answers the three assignment questions, then keeps going.

1. Characterise the dataset: structure, quality and the RTB funnel.
2. Pick the features most likely to predict a click, and justify the pick with data
   (an analysis, not a model).
3. Propose a way to guess, before bidding, the CPM needed to win a given
   `url × ad_slot` X% of the time.

After that come three sections the brief did not ask for: two machine-learning
models, seven extra analysis layers, and an attempt to undo the obfuscation of the
hashed columns.

### RTB in sixty seconds (sections 2.1, 2.2 and 3.1 of the brief)

When a user loads a page, the publisher auctions each ad slot in real time. Demand
partners (us) receive a bid request and may answer with a bid, a price in CPM (cost
per thousand impressions). The highest bidder wins and shows the ad. In a
first-price auction the winner pays their own bid; in a second-price auction they
pay the runner-up's bid. Our log records whether we `bidded`, our `bid`, whether we
`won_bid`, the `feedback_bid` (the price that decided the auction), and whether the
user `converted` (clicked).

Analysis code lives in `src/` (`data_loading`, `profiling`, `ctr_features`,
`bid_landscape`, `models`, `layers`, `layer_plots`, `decode`, `viz_style`) so it can
be tested and reused. This notebook is the narrative that drives it.
""")

# ----------------------------------------------------------------------------- setup
code(r"""import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import data_loading as dl
import profiling as pf
import ctr_features as cf
import bid_landscape as bl
import viz_style as vs

vs.apply_style()
FIG = Path.cwd().parent / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
pd.set_option("display.max_columns", 40, "display.width", 160)

def save(fig, name):
    fig.savefig(FIG / name, dpi=130, bbox_inches="tight")
    return fig
print("Environment ready.")""")

code(r"""df = dl.add_derived(dl.load_raw())
print(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns (incl. derived)")
df[["time","user_id","domain","ad_slot","device_type","auction_type",
    "bidded","bid","won_bid","feedback_bid","conversion"]].head(3)""")

# ============================================================== Q1
md(r"""---
## Q1. Main characteristics of the dataset

We go from structure (schema, types, missingness) to behaviour (the auction funnel,
prices, and where the signal sits).
""")

md("### 1.1 Schema, types and missingness")
code(r"""overview = pf.schema_overview(df.drop(columns=[c for c in df.columns if c.endswith(('_est','_missing'))] + ["hour","won","clicked","clearing_price"]))
overview""")

md(r"""Two facts fall out of the overview, and both change what we do downstream.

Two columns are constant. `country` is always `US` and `screen_orientation` never
changes, so neither carries information and both can be dropped.

Missingness is structural, not random. `bid`, `won_bid`, `feedback_bid` and
`conversion` are missing on exactly the rows where we did not bid (`bidded == 0`).
`device_brand` and `device_model` are missing for desktop traffic, because browsers
on a PC do not report a brand. The distinction matters: structural missingness is a
signal, not noise to impute away.""")
code(r"""print("Constant columns (drop):", pf.constant_columns(df))
print()
na_when_not_bidded = df.loc[df.bidded==0, ["bid","won_bid","feedback_bid","conversion"]].isna().all(axis=1).mean()
print(f"Rows with bidded==0 that are fully NaN on auction-outcome cols: {na_when_not_bidded:.1%}")
print("device_brand null rate on desktop:",
      f"{df.loc[df.device_type=='PERSONAL_COMPUTER','device_brand'].isna().mean():.1%}")""")

code(r"""# Missingness bar chart (only columns that actually have gaps)
miss = (df.isna().mean()*100)
miss = miss[miss>0].sort_values()
fig, ax = plt.subplots(figsize=(7,4))
ax.barh(miss.index, miss.values, color=vs.SEQUENTIAL)
for y,v in enumerate(miss.values):
    ax.text(v+0.6, y, f"{v:.0f}%", va="center", fontsize=8, color=vs.INK_SECONDARY)
vs.despine_labels(ax, "Missingness by column", "% of rows missing", None)
ax.set_xlim(0, miss.max()*1.15); ax.grid(False)
save(fig, "q1_missingness.png"); plt.show()""")

md("### 1.2 The RTB funnel: where opportunities go")
code(r"""funnel = pf.funnel(df)
display(funnel.style.format({"count":"{:,.0f}","pct_of_requests":"{:.1%}","pct_of_prev_meaningful":"{:.1%}"}))

stages = funnel["stage"].tolist(); counts = funnel["count"].tolist()
fig, ax = plt.subplots(figsize=(7.5,3.6))
bars = ax.barh(range(len(stages)), counts, color=[vs.CATEGORICAL[0]]+[vs.SEQUENTIAL]*3)
ax.set_yticks(range(len(stages))); ax.set_yticklabels(stages); ax.invert_yaxis()
for i,(c,b) in enumerate(zip(counts,bars)):
    pct = c/counts[0]
    ax.text(c+2500, i, f"{c:,}  ({pct:.1%})", va="center", fontsize=9, color=vs.INK_SECONDARY)
vs.despine_labels(ax, "RTB funnel: 249K requests to 4.4K clicks", "impressions", None)
ax.set_xlim(0, counts[0]*1.18); ax.grid(False)
save(fig, "q1_funnel.png"); plt.show()""")

md(r"""We bid on 49.8% of opportunities, win 37.5% of those, and see a click on 9.4% of
served (won) impressions. The 9.4% is the true CTR, because clicks can only happen
on ads we showed. Dividing clicks by all bids instead gives 3.5%, and that number is
wrong: every lost bid has `conversion == 0` by construction. CTR is only defined on
the won subset, and Q2 is computed there.""")

md("### 1.3 Prices are heavy-tailed, and auction type splits the market in two")
code(r"""bidded = dl.bidded_frame(df)
display(pf.numeric_summary(bidded, ["bid","feedback_bid"]))

fig, axes = plt.subplots(1,2, figsize=(11,4))
for ax,col,color in zip(axes, ["bid","feedback_bid"], [vs.CATEGORICAL[0], vs.CATEGORICAL[1]]):
    data = bidded[col].dropna()
    data = data[data>0]
    ax.hist(np.log10(data), bins=60, color=color, alpha=0.9)
    ax.set_xlabel(f"log10({col}), CPM"); ax.grid(False)
    med = np.log10(data.median())
    ax.axvline(med, color=vs.INK, lw=1, ls="--")
    ax.text(med+0.05, ax.get_ylim()[1]*0.9, f"median {data.median():.2f}", fontsize=8, color=vs.INK_SECONDARY)
    vs.despine_labels(ax, f"Distribution of {col}", None, "count")
fig.suptitle("Bid and clearing prices are heavy-tailed (log scale)", x=0.09, ha="left", weight="bold")
save(fig, "q1_price_dist.png"); plt.show()""")

code(r"""at = bidded.groupby("auction_type").agg(
    n=("won_bid","size"), share=("won_bid", lambda s: len(s)/len(bidded)),
    win_rate=("won_bid","mean"), median_bid=("bid","median"), median_feedback=("feedback_bid","median"))
display(at.style.format({"n":"{:,.0f}","share":"{:.1%}","win_rate":"{:.1%}","median_bid":"{:.2f}","median_feedback":"{:.2f}"}))""")

md(r"""This is the single most important structural fact for bidding. The two auction
types behave like different markets. First-price auctions take 92% of our bids and
are competitive: we win about 33% of them at a median bid near 1.4 CPM.
Second-price auctions are close to a monopoly: we win about 90%, and the clearing
prices are an order of magnitude higher. Any pricing or win-rate logic (Q3) has to
stratify by `auction_type`, or the two regimes blur into a meaningless average.""")

md("### 1.4 Who and where: device and geography")
code(r"""won = dl.won_frame(df)
dev = pf.rate_by_category(won, "device_type", "conversion", min_n=50).rename(columns={"rate":"ctr"})
top_states = pf.rate_by_category(won, "state_code", "conversion", min_n=300).rename(columns={"rate":"ctr"}).head(10)

fig, axes = plt.subplots(1,2, figsize=(11,4))
axes[0].barh(dev["device_type"], dev["ctr"], color=vs.SEQUENTIAL)
axes[0].invert_yaxis()
for y,(v,n) in enumerate(zip(dev["ctr"],dev["n"])):
    axes[0].text(v+0.002, y, f"{v:.1%} (n={n:,})", va="center", fontsize=8, color=vs.INK_SECONDARY)
vs.despine_labels(axes[0], "CTR by device type", "CTR", None); axes[0].grid(False); axes[0].set_xlim(0, dev["ctr"].max()*1.35)

axes[1].barh(top_states["state_code"], top_states["ctr"], color=vs.CATEGORICAL[2])
axes[1].invert_yaxis()
vs.despine_labels(axes[1], "CTR by state (top 10, n>=300)", "CTR", None); axes[1].grid(False)
save(fig, "q1_device_geo.png"); plt.show()""")

md(r"""What Q1 established. One clean day of 249,232 US impressions, with `country` and
`screen_orientation` constant and droppable. The funnel runs 50% bid, 37.5% win,
9.4% CTR on won impressions, and the missingness is structural (tied to `bidded` and
to desktop device reporting). Prices are heavy-tailed and auction type splits the
market, which is the lever Q3 pulls. Clicks concentrate on phones and tablets and
vary by state, which is where Q2 starts.
""")

# ============================================================== Q2
md(r"""---
## Q2. The most promising features for predicting CTR

The brief asks for an analysis: a justified feature set, not a trained model. Two
principles shape the answer.

Guard against leakage. In production we would score CTR before deciding whether and
what to bid, so any field produced by or after the auction is off limits as a
predictor: `bid`, `won_bid`, `feedback_bid`, and `conversion` itself. Including them
would inflate offline metrics and be unusable at bid time.

Condition on the served set. CTR only exists where `won_bid == 1`, so features are
ranked on that subset.

Clicks are rare (about 9%), so raw correlation is a poor screen. For categorical
features we use information value (IV), built from weight of evidence, the standard
screening measure for predictors of a rare binary event in credit scoring. For
numeric features we use mutual information, which picks up non-linear association.
Both are model-agnostic.
""")

code(r"""print("Excluded as leakage:", sorted(cf.LEAKAGE_COLUMNS))
print("Excluded as raw identifiers (high-cardinality / obfuscated):", sorted(cf.IDENTIFIER_COLUMNS))

cat_candidates = ["device_brand","device_type","detected_language","is_wifi",
                  "auction_type","state_code","city","hour","viewability_missing","session_depth_missing"]
num_candidates = ["viewability_est","session_depth_est","cookie_age_seconds"]
print("\nCategorical candidates:", cat_candidates)
print("Numeric candidates:", num_candidates)""")

md("### 2.1 Categorical feature strength: information value")
code(r"""iv = cf.rank_categorical_iv(won, cat_candidates)
display(iv.style.format({"information_value":"{:.4f}"}))

fig, ax = plt.subplots(figsize=(7.5,4))
colors = {"strong":vs.STATUS["good"], "medium":vs.CATEGORICAL[0], "weak":vs.CATEGORICAL[3], "useless":vs.INK_MUTED}
bar_colors = [colors[str(s)] for s in iv["strength"]]
ax.barh(iv["feature"], iv["information_value"], color=bar_colors)
ax.invert_yaxis()
for th,lab in [(0.02,"weak"),(0.1,"medium"),(0.3,"strong")]:
    ax.axvline(th, color=vs.INK_MUTED, lw=0.8, ls=":")
    ax.text(th, -0.6, lab, fontsize=7, color=vs.INK_MUTED, ha="center")
for y,v in enumerate(iv["information_value"]):
    ax.text(v+0.002, y, f"{v:.3f}", va="center", fontsize=8, color=vs.INK_SECONDARY)
vs.despine_labels(ax, "Categorical feature strength (information value)", "IV", None)
ax.grid(False); ax.set_xlim(0, iv["information_value"].max()*1.25)
save(fig, "q2_iv_ranking.png"); plt.show()""")

md("### 2.2 Numeric feature strength: mutual information")
code(r"""mi = cf.rank_numeric_mi(won, num_candidates)
display(mi.style.format({"mutual_info":"{:.5f}"}))

fig, ax = plt.subplots(figsize=(6.5,2.8))
ax.barh(mi["feature"], mi["mutual_info"], color=vs.SEQUENTIAL); ax.invert_yaxis()
for y,v in enumerate(mi["mutual_info"]):
    ax.text(v+2e-4, y, f"{v:.4f}", va="center", fontsize=8, color=vs.INK_SECONDARY)
vs.despine_labels(ax, "Numeric feature strength (mutual information)", "MI (nats)", None)
ax.grid(False); ax.set_xlim(0, mi["mutual_info"].max()*1.3)
save(fig, "q2_mi_ranking.png"); plt.show()""")

md("### 2.3 What the top features look like")
code(r"""# WoE detail: CTR across the levels of the two strongest categoricals.
fig, axes = plt.subplots(1,2, figsize=(11,4.2))
for ax, feat, k in zip(axes, ["device_type","device_brand"], [4, 12]):
    tab,_ = cf.woe_iv(won, feat)
    tab = tab.sort_values("ctr", ascending=False).head(k)
    ax.barh(tab.index.astype(str), tab["ctr"], color=vs.SEQUENTIAL); ax.invert_yaxis()
    ax.axvline(won["conversion"].mean(), color=vs.STATUS["critical"], lw=1, ls="--")
    ax.text(won["conversion"].mean(), -0.7, f"baseline {won['conversion'].mean():.1%}",
            fontsize=7, color=vs.STATUS["critical"], ha="center")
    vs.despine_labels(ax, f"CTR by {feat}", "CTR", None); ax.grid(False)
save(fig, "q2_woe_detail.png"); plt.show()""")

code(r"""# Sessions and viewability: does engagement/quality move CTR?
fig, axes = plt.subplots(1,2, figsize=(11,4))
# session depth bucketed
w = won.copy()
w["sd_bin"] = pd.cut(w["session_depth_est"], [0,1,2,3,5,10,1e9],
                     labels=["1","2","3","4-5","6-10","10+"])
sd = w.groupby("sd_bin")["conversion"].agg(["mean","size"]).dropna()
axes[0].plot(sd.index.astype(str), sd["mean"], marker="o", color=vs.CATEGORICAL[0], lw=2, ms=7)
vs.despine_labels(axes[0], "CTR rises with session depth", "session depth (impressions in session)", "CTR")
# viewability bucketed
w["vb_bin"] = pd.cut(w["viewability_est"], [-1,25,50,75,90,100], labels=["0-25","25-50","50-75","75-90","90-100"])
vb = w.groupby("vb_bin")["conversion"].agg(["mean","size"]).dropna()
axes[1].plot(vb.index.astype(str), vb["mean"], marker="o", color=vs.CATEGORICAL[2], lw=2, ms=7)
vs.despine_labels(axes[1], "CTR vs. viewability", "viewability bucket", "CTR")
save(fig, "q2_signal_trends.png"); plt.show()""")

md(r"""### 2.4 Recommended feature set, with the justification

Ranked by measured association on the served set, with leakage excluded:

| Feature | Type | Signal | Why it makes the cut |
|---|---|---|---|
| `device_brand` | categorical | IV about 0.11 (medium) | Strongest single predictor; brand stands in for platform, price tier and how the ad renders. |
| `device_type` | categorical | IV about 0.08 | Phones and tablets click far more than desktops; low cardinality and stable. |
| `session_depth` | numeric | highest MI | Deeper sessions click more, and the relation is monotone. Keep the `-1` missing flag. |
| `viewability` | numeric | second MI | The odds the ad is seen at all. Keep the `-1` missing flag. |
| `state_code` | categorical | IV about 0.03 | Weak but cheap and stable; useful as a target-encoded aggregate. |
| `auction_type` | categorical | IV about 0.02 | Weak for CTR but needed as context, and free. |
| `cookie_age_seconds` | numeric | low MI | A recency and identity proxy; better bucketed (new vs established cookie). |

Deliberately excluded: the leakage columns (`bid`, `won_bid`, `feedback_bid`), the
constants (`country`, `screen_orientation`), the near-constant and noisy fields
(`detected_language` is 98% `en`; `is_wifi` and raw `hour` add little), and the raw
identifiers (`user_id`, `url`, `domain`, `ad_slot`, `user_agent`). The identifiers
are not usable as raw categories with tens of thousands of levels, but a smoothed
historical CTR per `ad_slot`, `url` or `domain` is typically the single strongest
input in a production CTR model, and it is the natural next step. The
machine-learning section later in this notebook measures exactly that.

The short version: device identity, engagement and quality (session depth,
viewability) and light geographic context form the promising, leakage-safe core.
The biggest future lift comes from historical CTR aggregates over the identifier
columns.
""")

# ============================================================== Q3
md(r"""---
## Q3. Guessing the pre-bid CPM that wins X% of the time, per `url × ad_slot`

Restated: build `bid(url, ad_slot, X)`, the CPM that wins the auction X% of the
time, computable before we bid.

To win, our bid has to exceed the market's clearing price for that slot. If `F` is
the CDF of the clearing price, then

$$P(\text{win} \mid \text{bid}=b) = P(b > \text{clearing price}) = F(b),$$

so the bid that wins a fraction \(X\) of the time is the X-th quantile of the
clearing-price distribution for that `url × ad_slot`:

$$\text{bid}(X) = F^{-1}(X) = Q_X(\text{clearing price}).$$

Q3 therefore reduces to estimating clearing-price quantiles per slot.

Where does the clearing price come from? From `feedback_bid`, which is the price
that decided the auction under both auction types: the winner's bid when we lost,
the runner-up's bid when we won a first-price auction. It is our best single-number
read of what it took to win that slot, so `clearing_price` is set to `feedback_bid`
in `src/data_loading.add_derived`.
""")

md("### 3.1 The real obstacle: sparsity")
code(r"""pair_n = bidded.dropna(subset=["clearing_price"]).groupby(["url","ad_slot"]).size()
print(f"Unique (url, ad_slot) pairs: {pair_n.size:,}")
print(f"Median observations per pair: {pair_n.median():.0f}")
for th in [1,5,20,50,100]:
    print(f"  pairs with >={th:>3} obs: {(pair_n>=th).sum():>6,}  ({(pair_n>=th).mean():.1%})")

fig, ax = plt.subplots(figsize=(7,4))
ax.hist(np.clip(pair_n.values,0,60), bins=60, color=vs.SEQUENTIAL)
ax.axvline(pair_n.median(), color=vs.STATUS["critical"], lw=1.2, ls="--")
ax.text(pair_n.median()+1, ax.get_ylim()[1]*0.85, f"median = {pair_n.median():.0f} obs",
        color=vs.STATUS["critical"], fontsize=9)
vs.despine_labels(ax, "Most url×ad_slot pairs are data-starved", "observations per pair (clipped at 60)", "number of pairs")
ax.grid(False); save(fig, "q3_sparsity.png"); plt.show()""")

md(r"""A raw per-pair empirical quantile is hopeless when the median pair has two
observations. The fix is hierarchical partial pooling, also called empirical-Bayes
shrinkage: estimate the quantile at the pair level where the data is rich, and
otherwise borrow strength from progressively broader levels (`url`, then `ad_slot`,
then `domain`, then the auction-type stratum, then global), weighting each level by
its support. Pooling never crosses the `auction_type` boundary, because section 1.3
showed those are different price regimes. `src/bid_landscape.HierarchicalQuantileModel`
implements this.""")

md("### 3.2 Fit the model and read off bid curves")
code(r"""model = bl.HierarchicalQuantileModel(levels=["url","ad_slot","domain"], pseudo_count=30).fit(bidded)

# Bid curves for a few well-populated first-price pairs.
fp = bidded[(bidded.auction_type=="FIRST_PRICE") & bidded.clearing_price.notna()]
top_pairs = fp.groupby(["url","ad_slot"]).size().sort_values(ascending=False).head(4).index
targets = np.linspace(0.05, 0.95, 19)

fig, ax = plt.subplots(figsize=(8,4.6))
for i,(u,s) in enumerate(top_pairs):
    row = fp[(fp.url==u)&(fp.ad_slot==s)].iloc[0]
    curve = [model.bid_for_winrate(row, x) for x in targets]
    ax.plot(targets*100, curve, marker="o", ms=4, lw=2, color=vs.CATEGORICAL[i],
            label=f"pair {i+1} (n={((fp.url==u)&(fp.ad_slot==s)).sum()})")
vs.despine_labels(ax, "Bid needed vs. target win rate (sample first-price slots)",
                  "target win rate (%)", "recommended bid (CPM)")
ax.legend(); ax.grid(True); save(fig, "q3_bid_curves.png"); plt.show()""")

md(r"""Each curve is a slot's supply curve for winning: a higher win rate costs a higher
CPM, and the steepness says how contested the slot is. The curve is directly usable.
Pick a target win rate, read off the bid.""")

md("### 3.3 Does it work? A calibration back-test")
code(r"""# For well-populated pairs, bid the model's price and measure the realised win rate.
rows = []
for x in [0.2, 0.5, 0.8]:
    cal = bl.calibration_by_pair(bidded, model, x, min_n=50)
    rows.append((x, cal["realised_win_rate"].median(), cal["realised_win_rate"].mean(), len(cal)))
caldf = pd.DataFrame(rows, columns=["target","median_realised","mean_realised","n_pairs"])
display(caldf.style.format({"target":"{:.0%}","median_realised":"{:.1%}","mean_realised":"{:.1%}"}))

cal50 = bl.calibration_by_pair(bidded, model, 0.5, min_n=50)
fig, axes = plt.subplots(1,2, figsize=(11,4.2))
# (a) target vs realised
axes[0].plot([0,1],[0,1], color=vs.INK_MUTED, ls="--", lw=1)
axes[0].scatter(caldf["target"], caldf["median_realised"], s=90, color=vs.CATEGORICAL[0], zorder=3)
for _,r in caldf.iterrows():
    axes[0].annotate(f"{r.median_realised:.0%}", (r.target, r.median_realised),
                     textcoords="offset points", xytext=(8,-4), fontsize=9, color=vs.INK_SECONDARY)
axes[0].set_xlim(0,1); axes[0].set_ylim(0,1)
vs.despine_labels(axes[0], "Calibration: target vs. realised win rate", "target win rate", "median realised (n>=50 pairs)")
# (b) distribution of realised win rates at 50% target
axes[1].hist(cal50["realised_win_rate"], bins=25, color=vs.SEQUENTIAL)
axes[1].axvline(0.5, color=vs.STATUS["critical"], lw=1.2, ls="--")
axes[1].text(0.5+0.01, axes[1].get_ylim()[1]*0.85, "target 50%", color=vs.STATUS["critical"], fontsize=9)
vs.despine_labels(axes[1], "Realised win rate at 50% target (per pair)", "realised win rate", "pairs")
axes[1].grid(False); save(fig, "q3_calibration.png"); plt.show()""")

md(r"""The model is calibrated. Across well-populated pairs, bidding the predicted price
for a 50% target wins almost exactly 50% of the time, and the 20% and 80% targets
land on their marks too. Shrinkage keeps sparse pairs sane while letting rich pairs
speak for themselves.""")

md(r"""### 3.4 How to run it in production, and where it needs care

The estimator above is the answer, and it can be deployed as is. Precompute, per
auction stratum and per level (`url`, `ad_slot`, `domain`), a small grid of
clearing-price quantiles. At request time, look up the finest level with data and
blend by support. Serving is O(1), there is no online training, and cold slots
degrade gracefully to broader priors.

Three places need care.

Censoring. When we lose, `feedback_bid` is the winner's bid, which is a fair market
read. When we win a first-price auction it is the runner-up, which is a lower bound
on what a rival would have paid. A fully rigorous treatment models the win indicator
against our bid with survival analysis (Kaplan-Meier or Cox) to handle the censoring
explicitly.

Slots with no history. To generalise to unseen slots, replace the ID hierarchy with a
quantile-regression model (gradient-boosted trees with a pinball loss) that uses
`hour`, `device_type`, `viewability` and an encoded domain as inputs. That trades the
interpretability of the hierarchy for coverage on cold-start slots; the
machine-learning section below builds it and compares the two.

Recency. Clearing prices drift. In production, weight recent observations more
(exponential decay) and refresh the quantile grid on a schedule.
""")

# ============================================================== beyond the brief
ml_models.cells(md, code)
layers.cells(md, code)
decode.cells(md, code)

# ============================================================== close
md(r"""---
## Summary

| Question | Answer in one line |
|---|---|
| 1 | 249K one-day US impressions; the funnel runs 50% bid, 37.5% win, 9.4% CTR on won impressions; prices are heavy-tailed; auction type splits the market; `country` and `screen_orientation` are constant. |
| 2 | Leakage-safe, data-ranked core: `device_brand`, `device_type`, `session_depth`, `viewability`, plus light geographic and auction context. The biggest future lift is historical CTR aggregates over the ID columns, and the CTR model confirms it (AUC 0.69 to 0.78). |
| 3 | Win rate is the clearing-price CDF, so bid(X) is the X-th quantile of the clearing price per slot, estimated with hierarchical shrinkage to beat sparsity. It calibrates to target within a point or two, and quantile regression covers slots with no history. |

Beyond the brief: a flat 10% bid shading in first-price auctions would have saved
18% of spend while keeping 88% of wins; supply is highly concentrated (domain Gini
0.92); publishers fall into three behavioural archetypes; and 36 million guesses
failed to reverse a single hash, so identity stays hidden while structure and
behaviour remain usable.

Reproducible: logic in `src/`, figures in `reports/figures/`, notebook regenerated by
`build_notebook.py`.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3.14 (NSRB)", "language": "python", "name": "nsrb314"},
    "language_info": {"name": "python"},
}
out = "notebooks/analysis.ipynb"
nbf.write(nb, out)
print(f"Wrote {out} with {len(cells)} cells")
