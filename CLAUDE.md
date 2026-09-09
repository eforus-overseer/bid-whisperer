# CLAUDE.md

Guidance for AI assistants (and humans) working in this repository.

## What this project is

An end-to-end analysis of a **real-time-bidding (RTB) ad-impression dataset** from a
take-home assignment. One day (2021-02-01), US-only, **249,232 rows × 26
columns**, one auction opportunity per row. The repo answers three questions:

1. **Q1 — Characterise the dataset** (structure, quality, the RTB funnel).
2. **Q2 — CTR feature selection** (a data-justified feature set; *analysis*, not a model).
3. **Q3 — Pre-bid pricing** (guess the CPM to win a `url × ad_slot` X% of the time).

The primary deliverable is `notebooks/analysis.ipynb`; the analysis logic lives in
`src/` so it stays testable and reusable.

## Architecture & where things live

| Path | Responsibility |
|---|---|
| `src/data_loading.py` | Load CSV with correct dtypes; parse UTC time; add derived columns (`clearing_price`, sentinel-aware `*_est`/`*_missing`, `won`/`clicked`). `bidded_frame()` / `won_frame()` subset helpers. |
| `src/profiling.py` | Q1 tidy-table builders: `schema_overview`, `constant_columns`, `funnel`, `rate_by_category`, `numeric_summary`. No plotting (side-effect free). |
| `src/ctr_features.py` | Q2: `woe_iv`, `rank_categorical_iv` (Information Value), `rank_numeric_mi` (mutual information). Holds `LEAKAGE_COLUMNS` and `IDENTIFIER_COLUMNS`. |
| `src/bid_landscape.py` | Q3: `HierarchicalQuantileModel` (empirical-Bayes shrinkage over `url→ad_slot→domain`, stratified by `auction_type`), plus `empirical_winrate_curve` and `calibration_by_pair` for back-testing. |
| `src/viz_style.py` | Matplotlib style + validated colorblind-safe palette. Call `apply_style()` once. |
| `build_notebook.py` | Generates `notebooks/analysis.ipynb` from Python (narrative is diffable). |

## Domain facts that must not be forgotten (they encode real bugs if ignored)

- **CTR is only defined on won impressions.** `conversion` is structurally `0` for every
  lost bid. Compute CTR on `won_frame(df)`, never over all bids (that deflates it to ~3.5%
  instead of the true ~9.4%).
- **Auction-outcome columns are leakage for CTR.** `bid`, `won_bid`, `feedback_bid` are
  produced by/after the auction — never use them to predict clicks.
- **Missingness is structural.** Auction-outcome columns are NaN exactly where
  `bidded == 0`; `device_brand`/`device_model` are NaN for desktop. Don't blindly impute.
- **`country` and `screen_orientation` are constants** — zero predictive value.
- **Stratify by `auction_type` for anything price-related.** First-price and second-price
  are effectively different markets (win ~33% @ ~1.4 CPM vs. ~90% @ ~38 CPM).
- **Sentinels:** `viewability == -1` and `session_depth == -1` mean "could not estimate,"
  not a real value. Use the `*_est` columns for stats and keep the `*_missing` flag.
- **Q3 core identity:** `P(win | bid) = CDF(clearing_price)` ⇒ `bid(X) = quantile_X(clearing_price)`.
  `clearing_price ≈ feedback_bid` (the price that decided the auction).

## Conventions

- Python ≥ 3.11 (developed on 3.14). Deps in `requirements.txt`.
- Analysis functions return **tidy DataFrames**; plotting stays in the notebook.
- Import from `src/` by adding it to `sys.path` (the notebook does this in its setup cell).
- Charts: call `viz_style.apply_style()`, assign categorical hues in fixed order (never
  cycle), one y-axis per chart, single-hue sequential ramps. Palette is CVD-validated.

## Working commands

```bash
python3 build_notebook.py                                   # regenerate notebook source
python3 -m jupyter nbconvert --to notebook --execute \      # execute end-to-end
  --inplace --ExecutePreprocessor.kernel_name=nsrb314 notebooks/analysis.ipynb
```

## Data & privacy

The dataset (`data/data.csv`) and the assignment brief (`Analysis-NSRB.pdf`) are the
assignment author's property and are **git-ignored**. Do not commit them or paste raw rows
into public artifacts. The repo ships the method, not the data. Fetch instructions are in
the README.

## If you extend this

- Q2 → build the actual CTR model; the highest-value new feature is a **smoothed historical
  CTR aggregate** per `ad_slot` / `url` / `domain`.
- Q3 → add **survival analysis** (Kaplan–Meier/Cox) for censored losses, and a
  **quantile-regression** model (GBM with pinball loss) for cold-start slots unseen in the
  ID hierarchy.
- Add `tests/` for the `src/` functions (funnel counts, IV monotonicity, calibration).
