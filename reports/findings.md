# Findings

Companion to `notebooks/analysis.ipynb`. All numbers are computed from the raw
249,232-row dataset (2021-02-01, US). The machine-readable versions are
`reports/metrics_models.json` and `reports/metrics_layers.json`.

## Q1. Dataset characteristics

Shape and scope. 249,232 impressions and 26 columns, one calendar day, `country`
uniformly `US`. Each row is one auction opportunity.

Data quality. `country` and `screen_orientation` are constant and can be dropped.
Missingness is structural: `bid`, `won_bid`, `feedback_bid` and `conversion` are
NaN on exactly the 125,080 rows where `bidded == 0`, and `device_brand` and
`device_model` are about 55% null, almost entirely on desktop, where browsers do not
report a brand. `viewability == -1` and `session_depth == -1` are sentinels for
"could not estimate".

The funnel (`reports/figures/q1_funnel.png`, `reports/figures/diagram_rtb-funnel.png`):

| Stage | Count | Share of requests | Share of previous stage |
|---|--:|--:|--:|
| Requests | 249,232 | 100% | |
| Bidded | 124,152 | 49.8% | 49.8% |
| Won | 46,562 | 18.7% | 37.5% |
| Clicked | 4,368 | 1.8% | 9.4% |

The true CTR is 9.4%, measured on served (won) impressions only. Dividing clicks by
all bids gives 3.5%, which is wrong because every lost bid has `conversion = 0` by
construction.

Prices and auction type (`q1_price_dist.png`). Bid and clearing prices are strongly
right-skewed (median bid about 0.74 CPM, 99th percentile about 79). Auction type
splits the market:

| auction_type | Share of bids | Win rate | Median bid |
|---|--:|--:|--:|
| FIRST_PRICE | 92.5% | 33.2% | 1.39 |
| SECOND_PRICE | 7.5% | 90.4% | 38.1 |

Device and geography (`q1_device_geo.png`). CTR is highest on tablets (about 13%)
and high-end phones (about 12%), lowest on desktop (about 7%), and varies by state.

## Q2. Promising CTR features

Ranked on the won subset, excluding leakage (`bid`, `won_bid`, `feedback_bid`) and
raw identifiers. Categorical strength by information value; numeric strength by
mutual information.

| Feature | Measure | Value | Reading |
|---|---|--:|---|
| `device_brand` | IV | about 0.11 | medium; the strongest single predictor |
| `device_type` | IV | about 0.08 | weak to medium; low cardinality, stable |
| `session_depth` | MI | highest | engagement proxy; CTR rises monotonically |
| `viewability` | MI | second | the chance the ad is seen at all |
| `state_code` | IV | about 0.03 | weak but cheap, stable geographic context |
| `auction_type` | IV | about 0.02 | weak for CTR, needed as context |
| `cookie_age_seconds` | MI | low | bucket it: new against established |

Recommended core: `device_brand`, `device_type`, `session_depth`, `viewability`,
plus light geographic and auction context. Excluded: the constants, the
near-constants (`detected_language` is 98% `en`) and the leakage columns. The
biggest future lift is a smoothed historical CTR per `ad_slot`, `url` or `domain`;
the CTR model below measures it.

Figures: `q2_iv_ranking.png`, `q2_mi_ranking.png`, `q2_woe_detail.png`,
`q2_signal_trends.png`.

## Q3. Pre-bid CPM to win X% of the time

Method. To win, the bid must exceed the market clearing price, so
`P(win | bid) = CDF(clearing_price)` and bid(X) is the X-th quantile of the
clearing-price distribution for that `url × ad_slot`. `clearing_price` is
`feedback_bid`, the price that decided the auction under both auction types. The
problem reduces to quantile estimation per slot.

Obstacle (`q3_sparsity.png`). 13,817 unique `url × ad_slot` pairs with a median of
two observations each; only 168 pairs have 100 or more. A raw per-pair quantile is
noise.

Solution. Hierarchical partial pooling (empirical-Bayes shrinkage): blend the
pair-level quantile toward progressively broader levels (`url`, `ad_slot`, `domain`,
global) with weight `n / (n + 30)`, stratified by `auction_type` so the two price
regimes never pool. `reports/figures/diagram_bid-engine.png` shows the serving path.

Validation (`q3_calibration.png`). Bidding the model's price on pairs with at least
50 observations:

| Target | Median realised win rate |
|--:|--:|
| 20% | about 20% |
| 50% | 50.5% |
| 80% | about 80% |

Production. Precompute a small per-level quantile grid for an O(1) lookup at request
time with a graceful cold-start fallback. Weight recent data, since prices drift.
Extend with survival analysis for censored losses and with the quantile regressor
below for slots absent from the hierarchy.

## Beyond the brief: machine-learning models

CTR classifier (`ml_ctr_roc_pr.png`, `ml_ctr_calibration.png`,
`ml_ctr_importance.png`). Five-fold stratified cross-validation on the 46,562 served
impressions, base rate 9.4%, no auction outcomes as inputs, all identifier encodings
out of fold.

| Model | AUC | Average precision | Log loss |
|---|--:|--:|--:|
| Gradient boosting, context plus ID encodings | 0.780 | 0.257 | 0.265 |
| Gradient boosting, context only | 0.687 | 0.167 | 0.289 |
| Logistic regression, context only | 0.675 | 0.158 | 0.293 |

The identifier encodings add 0.09 AUC. Permutation importance: viewability (0.070),
the ad-slot encoding (0.068), the url encoding (0.029), the publisher-properties
encoding (0.028), then the domain encoding and session depth. Device brand drops
down the list because device type and operating system carry the same information.

Quantile regression on unseen pairs (`ml_qr_coverage.png`, `ml_qr_pinball.png`).
Grouped 80/20 split by `(url, ad_slot)`: 23,973 test rows in 2,764 pairs never seen
in training.

| Quantile | Coverage, regressor | Coverage, lookup | Pinball (CPM), regressor | Pinball (CPM), lookup |
|--:|--:|--:|--:|--:|
| 0.2 | 0.278 | 0.255 | 0.961 | 0.968 |
| 0.5 | 0.541 | 0.522 | 2.014 | 2.122 |
| 0.8 | 0.804 | 0.798 | 2.468 | 2.495 |

Both are close to target on cold slots; the regressor wins on the proper score at
every quantile. Recommendation: the lookup for slots with history, the regressor for
new pairs.

## Beyond the brief: extra layers

Bid shading (`layer_overpay.png`). On 38,167 won first-price auctions the median
overpay is 41% of our bid (interquartile range 20% to 64%); 39.9% of wins paid more
than twice the runner-up. Runner-up prices sum to 53% of spend, so the ceiling on
savings is 47.3%. A flat 10% shading keeps 88.0% of wins, saves 18.4% of spend and
cuts the cost per won impression from 2.08 to 1.92 CPM. Lost first-price auctions
are lost by a median 0.99 CPM; only 4.2% were within 10% of our bid.

Hourly profile (`layer_hourly.png`). Requests range from 16,366 at 17 UTC to 3,225
at 09 UTC. Bid rate falls with volume (56% at 01 UTC, 38% at 08 UTC). Win rate peaks
at 47% at 04 UTC and bottoms at 28% at 08 to 09 UTC. Median bid climbs from 0.57 CPM
at 00 UTC to 0.93 at 17 UTC. Hourly CTR moves between 7.6% and 12.0% on small counts.

Concentration (`layer_lorenz.png`). Gini 0.918 for domains, 0.786 for urls, 0.814
for ad slots. The top 10 domains carry 23.6% of impressions, the top 50 carry 50.0%,
the top 500 carry 86.5%.

Browser and OS (`layer_browser_os.png`). Chrome is 54.8% of impressions at 9.9% CTR,
Safari 14.8% at 9.9%, Edge 12.7% at 8.1%, Firefox 4.7% at 5.7%. By operating system:
Android 14.1% CTR, iOS 9.7%, macOS 8.1%, Windows 7.0%.

Cookie age (`layer_cookie_age.png`). New cookies are half of served impressions at
9.0% CTR; under one day 7.3%; one to seven days 8.0%; then monotone up to 10.5% for
cookies older than a year.

Publisher archetypes (`layer_archetypes.png`). KMeans on standardised behavioural
fingerprints of the 294 domains with at least 100 impressions, k = 3 by silhouette
(0.227). Desktop publishers: 178 domains, 4% mobile, median session depth 27.7, CTR
6.9%. Mobile publishers: 95 domains, 89% mobile, 84% wifi, CTR 12.9%. Second-price
premium: 21 domains, 96% second-price, median bid 24.1 CPM, win rate 88%, CTR 14.2%.
172 domains are desktop-only and 56 mobile-only; the top five domains hold 25% of
second-price impressions.

Geography (`layer_state_map.png`). Among states with at least 300 served
impressions, Maryland (12.3%), Kentucky (12.0%), Alabama (11.4%) and Iowa (11.0%)
click most; Louisiana (6.5%), Utah (7.1%), Massachusetts (7.2%) and Arkansas (7.5%)
least. State is missing on 3.4% of rows, and part of the spread is device mix.

Users. `user_id` is unique on all 249,232 rows; frequency and reach cannot be
studied in this sample.

## Beyond the brief: can the obfuscation be undone?

Format (`decode_nibbles.png`). `domain`, `url`, `ad_slot` and
`publisher_properties` are all 32 lowercase hex characters with first-digit shares
between 5.6% and 6.9% (uniform is 6.25%), the profile of an MD5-style digest.
`user_id` matches UUID version 4 on every row.

Attacks. Dictionary: the Tranco top one million domains, nine spellings, four digest
functions, 36,000,000 digests, zero hits. Integers: md5 of 0 to 3,000,000, single
and double, zero hits. Chaining: each hashed column hashed into the others, zero
hits. The hashes are salted or keyed; identity cannot be recovered.

Structure (`decode_fd_matrix.png`, `diagram_hash-schema.png`). Functional
dependencies survive hashing: `url` determines `domain` (1.000), `ad_slot`
determines `domain` (0.981) and `url` (0.786), `publisher_properties` determines
`domain` (0.870), `domain` determines `publisher_properties` (0.791). A page belongs
to one domain, a slot is a site-level placement reused across pages, and publisher
properties sit above the domain like a seller or deal id.
