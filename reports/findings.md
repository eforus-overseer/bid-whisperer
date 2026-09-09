# NSRB RTB Analysis — Findings Report

*Companion to `notebooks/analysis.ipynb`. All numbers computed from the raw 249,232-row
dataset (2021-02-01, US).*

---

## Q1 — Dataset characteristics

**Shape & scope.** 249,232 impressions × 26 columns, one calendar day, `country` is
uniformly `US`. Each row is a single auction opportunity.

**Data quality.**
- **Constant columns:** `country`, `screen_orientation` — drop them (zero variance).
- **Structural missingness** (not random, therefore informative):
  - `bid`, `won_bid`, `feedback_bid`, `conversion` are NaN on **exactly** the 125,080 rows
    where `bidded == 0`.
  - `device_brand` / `device_model` are ~55% null, concentrated on desktop
    (`PERSONAL_COMPUTER`), where browsers don't report a brand.
- **Sentinels:** `viewability == -1` and `session_depth == -1` mean "could not estimate."

**The funnel** (`reports/figures/q1_funnel.png`):

| Stage | Count | % of requests | % of previous |
|---|--:|--:|--:|
| Requests | 249,232 | 100% | — |
| Bidded | 124,152 | 49.8% | 49.8% |
| Won | 46,562 | 18.7% | **37.5%** |
| Converted (click) | 4,368 | 1.8% | **9.4%** |

> **True CTR = 9.4%**, measured on *served (won)* impressions only. Dividing clicks by all
> bids gives a misleading 3.5%, because every lost bid has `conversion = 0` by construction.

**Prices & auction type** (`q1_price_dist.png`). Bid and clearing prices are strongly
right-skewed (median bid ≈ 0.74 CPM, 99th pct ≈ 79). Auction type splits the market:

| auction_type | share of bids | win rate | median bid |
|---|--:|--:|--:|
| FIRST_PRICE | 92.5% | 33.2% | 1.39 |
| SECOND_PRICE | 7.5% | **90.4%** | **38.1** |

**Engagement geography** (`q1_device_geo.png`). CTR is highest on TABLET (~13%) and
HIGHEND_PHONE (~12%), lowest on desktop (~7%), and varies by state.

---

## Q2 — Promising CTR features

Ranked on the **won** subset, excluding leakage (`bid`, `won_bid`, `feedback_bid`) and raw
identifiers. Categorical strength by **Information Value**; numeric by **mutual information**.

| Feature | Measure | Value | Verdict |
|---|---|--:|---|
| `device_brand` | IV | ~0.11 | **medium — strongest single predictor** |
| `device_type` | IV | ~0.08 | weak-medium; robust, low-cardinality |
| `session_depth` | MI | highest | engagement proxy, monotonic ↑ |
| `viewability` | MI | 2nd | probability the ad is actually seen |
| `state_code` | IV | ~0.03 | weak but cheap/stable geo context |
| `auction_type` | IV | ~0.02 | weak for CTR; essential context |
| `cookie_age_seconds` | MI | low | bucket as new-vs-established |

**Recommended core set:** `device_brand`, `device_type`, `session_depth`, `viewability`,
plus light geo/auction context. **Excluded:** constants (`country`, `screen_orientation`),
near-constants (`detected_language` — 98% `en`), and leakage columns. **Biggest future
lift:** smoothed **historical-CTR aggregates** over the identifier columns (`ad_slot`,
`url`, `domain`) — typically the dominant feature in production RTB CTR models.

See `q2_iv_ranking.png`, `q2_mi_ranking.png`, `q2_woe_detail.png`, `q2_signal_trends.png`.

---

## Q3 — Pre-bid CPM to win X% of the time

**Method.** To win, the bid must exceed the market **clearing price**; hence
`P(win | bid) = CDF(clearing_price)` and

> **bid(X) = X-th quantile of the clearing-price distribution for that `url × ad_slot`.**

`clearing_price ≈ feedback_bid` (the price that decided the auction under both auction
types). The problem reduces to **quantile estimation per slot.**

**Obstacle: sparsity** (`q3_sparsity.png`). 13,817 unique `url × ad_slot` pairs but a
**median of 2 observations each**; only 168 pairs have ≥100. A raw per-pair quantile is
uselessly noisy.

**Solution: hierarchical partial pooling** (empirical-Bayes shrinkage). Blend the pair-level
quantile toward progressively broader levels (`url → ad_slot → domain → global`), weighted
by support, **stratified by `auction_type`** (never pooling across the two price regimes).

**Validation** (`q3_calibration.png`). Back-testing on well-populated pairs, bidding the
model's price achieves the target win rate almost exactly:

| Target | Median realised (n ≥ 50 pairs) |
|--:|--:|
| 20% | ~20% |
| 50% | ~50% |
| 80% | ~80% |

**Productionisation.** Precompute a small per-level quantile grid → O(1) lookup at request
time, graceful cold-start fallback. Extend with **survival analysis** (Kaplan–Meier/Cox) to
handle censored losses rigorously, and **quantile regression** (GBM with pinball loss) to
generalise to slots absent from the ID hierarchy. Weight recent data (prices drift).
