"""Machine-learning layer for the RTB impression analysis.

Two experiments that go past the original brief:

1. A click-through-rate classifier trained on the impressions we actually served.
   Its job is to test the Q2 feature analysis with a model, and to measure how
   much the hashed identifiers (domain, url, ad_slot, publisher_properties) add
   once they are turned into out-of-fold "historical CTR" encodings.

2. Quantile regression of the clearing price, trained and tested on disjoint
   sets of (url, ad_slot) pairs. This is the cold-start case: a slot the
   hierarchical lookup in `bid_landscape` has never seen, so the lookup can only
   fall back to domain or global quantiles. The regressor can instead use the
   request's context (device, hour, viewability, state, browser) to place the bid.

Leakage rules are the same as in `ctr_features`: nothing produced by or after the
auction (bid, won_bid, feedback_bid, clearing_price) is a CTR feature.

All target encodings are out-of-fold. Inside a training set, each row's encoding
is computed from the other inner folds, so no row ever sees its own label. Rows
outside the training set are encoded with the full training-set statistics.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_pinball_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

import bid_landscape as bl
import viz_style as vs

# ----------------------------------------------------------------------------- feature lists
LEAKAGE_COLUMNS = {"bid", "won_bid", "feedback_bid", "conversion", "clicked", "won", "clearing_price"}

# Low-cardinality categoricals. Each has fewer than 255 levels, which is what the
# histogram booster needs for native categorical splits.
CATEGORICAL = ["device_brand", "device_type", "state_code", "hour", "is_wifi", "auction_type", "browser", "os"]

# Numeric columns. NaN is left in place for the booster (it learns a missing branch);
# the logistic baseline imputes the median.
CTR_NUMERIC = ["viewability_est", "viewability_missing", "session_depth_est", "session_depth_missing", "log_cookie_age"]
QR_NUMERIC = ["viewability_est", "viewability_missing", "session_depth_est", "session_depth_missing"]

# Hashed identifiers that only become usable through target encoding.
CTR_ID_COLUMNS = ["domain", "url", "ad_slot", "publisher_properties"]
QR_ID_COLUMNS = ["domain", "url", "ad_slot"]

NA_TOKEN = "__na__"


# ----------------------------------------------------------------------------- feature building
def parse_user_agent(ua: pd.Series) -> pd.DataFrame:
    """Browser and OS family from the raw user-agent string.

    Order matters: Edge and Opera embed "Chrome/", and Chrome embeds "Safari/",
    so the more specific tokens are tested first.
    """
    s = ua.fillna("").astype(str)
    browser = np.select(
        [
            s.str.contains("Edg/", regex=False),
            s.str.contains("OPR/|Opera", regex=True),
            s.str.contains("Firefox/", regex=False),
            s.str.contains("Chrome/", regex=False),
            s.str.contains("Safari/", regex=False),
        ],
        ["Edge", "Opera", "Firefox", "Chrome", "Safari"],
        default="Other",
    )
    os_family = np.select(
        [
            s.str.contains("Windows", regex=False),
            s.str.contains("iPhone|iPad", regex=True),
            s.str.contains("Android", regex=False),
            s.str.contains("Macintosh", regex=False),
            s.str.contains("CrOS", regex=False),
            s.str.contains("Linux", regex=False),
        ],
        ["Windows", "iOS", "Android", "macOS", "ChromeOS", "Linux"],
        default="Other",
    )
    return pd.DataFrame({"browser": browser, "os": os_family}, index=ua.index)


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Assemble model inputs from a frame produced by `data_loading.add_derived`.

    Returns a copy with parsed user-agent families, a log cookie age, string
    categoricals with a shared NA token, and the identifier columns filled so they
    can be used as encoding keys. Auction-outcome columns are dropped.
    """
    out = df.copy()
    out[["browser", "os"]] = parse_user_agent(out["user_agent"])
    out["log_cookie_age"] = np.log1p(out["cookie_age_seconds"].clip(lower=0))
    for c in CATEGORICAL + CTR_ID_COLUMNS:
        out[c] = out[c].astype(object).where(out[c].notna(), NA_TOKEN).astype(str)
    for c in ["viewability_missing", "session_depth_missing"]:
        out[c] = out[c].astype(int)
    out["viewability_est"] = out["viewability_est"].astype(float)
    out["session_depth_est"] = out["session_depth_est"].astype(float)
    return out.drop(columns=[c for c in LEAKAGE_COLUMNS if c in out.columns and c not in ("conversion", "clearing_price")])


# ----------------------------------------------------------------------------- target encoding
def _smoothed_means(keys: pd.Series, y: np.ndarray, prior: float | pd.Series, m: float) -> pd.Series:
    """Per-key smoothed mean of y: (sum_y + m * prior) / (n + m).

    `prior` is either a scalar or a Series indexed like `keys` (per-row prior, used
    for stratified encodings).
    """
    frame = pd.DataFrame({"k": keys.to_numpy(), "y": y})
    if np.isscalar(prior):
        frame["p"] = prior
    else:
        frame["p"] = np.asarray(prior)
    g = frame.groupby("k", sort=False).agg(s=("y", "sum"), n=("y", "size"), p=("p", "first"))
    return (g["s"] + m * g["p"]) / (g["n"] + m)


class TargetEncoder:
    """Out-of-fold smoothed target encoding for high-cardinality keys.

    Parameters
    ----------
    cols : list[str]
        Key columns to encode. Each produces one feature named ``te_<col>``.
    m : float
        Smoothing strength in pseudo-observations. A key with n rows gets weight
        n / (n + m) on its own mean and m / (n + m) on the prior.
    strata_col : str | None
        If set, keys are formed within strata (e.g. auction_type) and the prior is
        the stratum mean instead of the global mean. This keeps first-price and
        second-price prices from blending into one encoding.
    n_splits : int
        Inner folds for the out-of-fold encoding of training rows.
    """

    def __init__(self, cols: list[str], m: float = 20.0, strata_col: str | None = None, n_splits: int = 5, seed: int = 0):
        self.cols = cols
        self.m = m
        self.strata_col = strata_col
        self.n_splits = n_splits
        self.seed = seed
        self.maps_: dict[str, pd.Series] = {}
        self.prior_: float | pd.Series | None = None

    # -- helpers -------------------------------------------------------------
    def _keys(self, X: pd.DataFrame, col: str) -> pd.Series:
        if self.strata_col is None:
            return X[col].astype(str)
        return X[self.strata_col].astype(str) + "|" + X[col].astype(str)

    def _prior_for(self, X: pd.DataFrame, y: np.ndarray | None) -> float | pd.Series:
        """Prior per row: global mean, or stratum mean when stratified."""
        if self.strata_col is None:
            return float(np.mean(y)) if y is not None else float(self.prior_)
        if y is not None:
            strata_means = pd.Series(y).groupby(X[self.strata_col].astype(str).to_numpy()).mean()
            self.strata_means_ = strata_means
            self.global_mean_ = float(np.mean(y))
        mapped = X[self.strata_col].astype(str).map(self.strata_means_)
        return mapped.fillna(self.global_mean_)

    # -- API -----------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "TargetEncoder":
        y = np.asarray(y, dtype=float)
        prior = self._prior_for(X, y)
        self.prior_ = prior if np.isscalar(prior) else float(np.mean(y))
        self.maps_ = {c: _smoothed_means(self._keys(X, c), y, prior, self.m) for c in self.cols}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        prior = self._prior_for(X, None)
        out = {}
        for c in self.cols:
            enc = self._keys(X, c).map(self.maps_[c])
            out[f"te_{c}"] = enc.fillna(pd.Series(prior, index=X.index) if not np.isscalar(prior) else prior).to_numpy()
        return pd.DataFrame(out, index=X.index)

    def fit_transform_oof(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """Encode training rows out-of-fold, then fit on all rows for later transform."""
        y = np.asarray(y, dtype=float)
        oof = pd.DataFrame(index=X.index, columns=[f"te_{c}" for c in self.cols], dtype=float)
        kf = KFold(self.n_splits, shuffle=True, random_state=self.seed)
        for fit_idx, enc_idx in kf.split(X):
            inner = TargetEncoder(self.cols, self.m, self.strata_col, self.n_splits, self.seed)
            inner.fit(X.iloc[fit_idx], y[fit_idx])
            oof.iloc[enc_idx] = inner.transform(X.iloc[enc_idx]).to_numpy()
        self.fit(X, y)
        return oof


# ----------------------------------------------------------------------------- matrix assembly
class BoosterMatrix:
    """Turn a feature frame into the numeric matrix the histogram booster wants.

    Categoricals become integer codes with -1 for unseen or missing (the booster
    treats negative codes as missing). Numerics pass through with NaN intact.
    """

    def __init__(self, categorical: list[str], numeric: list[str], extra: list[str] | None = None):
        self.categorical = categorical
        self.numeric = numeric
        self.extra = extra or []
        self.enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1)

    @property
    def columns(self) -> list[str]:
        return self.categorical + self.numeric + self.extra

    @property
    def categorical_mask(self) -> list[bool]:
        return [True] * len(self.categorical) + [False] * (len(self.numeric) + len(self.extra))

    def fit(self, X: pd.DataFrame) -> "BoosterMatrix":
        self.enc.fit(X[self.categorical].astype(str))
        return self

    def transform(self, X: pd.DataFrame, extra: pd.DataFrame | None = None) -> pd.DataFrame:
        codes = self.enc.transform(X[self.categorical].astype(str))
        mat = pd.DataFrame(codes, columns=self.categorical, index=X.index)
        for c in self.numeric:
            mat[c] = X[c].to_numpy(dtype=float)
        if extra is not None:
            for c in self.extra:
                mat[c] = extra[c].to_numpy(dtype=float)
        return mat[self.columns]


def _hgb_classifier(seed: int, categorical_mask: list[bool]) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        categorical_features=categorical_mask,
        learning_rate=0.06,
        max_iter=400,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=seed,
    )


def _logreg_baseline(categorical: list[str], numeric: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), categorical),
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), numeric),
        ]
    )
    return Pipeline([("pre", pre), ("lr", LogisticRegression(max_iter=3000, C=1.0))])


def _binary_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
        "brier": float(brier_score_loss(y, p)),
    }


# ----------------------------------------------------------------------------- experiment 1: CTR
def run_ctr_experiment(
    won: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 0,
    m: float = 20.0,
    importance_rows: int = 10_000,
    importance_repeats: int = 5,
) -> dict:
    """Cross-validated CTR models on the served impressions.

    Three variants share the same outer folds:
      hgb_full   gradient boosting on context features plus out-of-fold target
                 encodings of domain, url, ad_slot, publisher_properties
      hgb_no_id  the same booster without the identifier encodings (ablation)
      logreg     logistic regression on one-hot categoricals plus scaled numerics

    Returns a dict with pooled out-of-fold predictions, pooled and per-fold
    metrics, the permutation importance from the first fold, and curve data.
    """
    t0 = time.time()
    X = build_feature_frame(won)
    y = won["conversion"].astype(int).to_numpy()
    te_cols = [f"te_{c}" for c in CTR_ID_COLUMNS]

    names = ["hgb_full", "hgb_no_id", "logreg"]
    oof = {n: np.zeros(len(y)) for n in names}
    fold_rows = []
    importance = None

    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    for k, (tr, va) in enumerate(skf.split(X, y)):
        Xtr, Xva, ytr, yva = X.iloc[tr], X.iloc[va], y[tr], y[va]

        encoder = TargetEncoder(CTR_ID_COLUMNS, m=m, n_splits=n_splits, seed=seed)
        te_tr = encoder.fit_transform_oof(Xtr, ytr)
        te_va = encoder.transform(Xva)

        full = BoosterMatrix(CATEGORICAL, CTR_NUMERIC, te_cols).fit(Xtr)
        Mtr_full, Mva_full = full.transform(Xtr, te_tr), full.transform(Xva, te_va)
        clf_full = _hgb_classifier(seed, full.categorical_mask).fit(Mtr_full, ytr)
        oof["hgb_full"][va] = clf_full.predict_proba(Mva_full)[:, 1]

        lean = BoosterMatrix(CATEGORICAL, CTR_NUMERIC).fit(Xtr)
        clf_lean = _hgb_classifier(seed, lean.categorical_mask).fit(lean.transform(Xtr), ytr)
        oof["hgb_no_id"][va] = clf_lean.predict_proba(lean.transform(Xva))[:, 1]

        lr = _logreg_baseline(CATEGORICAL, CTR_NUMERIC).fit(Xtr, ytr)
        oof["logreg"][va] = lr.predict_proba(Xva)[:, 1]

        for n in names:
            fold_rows.append({"fold": k, "model": n, **_binary_metrics(yva, oof[n][va])})

        if k == 0:
            rng = np.random.RandomState(seed)
            sub = rng.choice(len(va), size=min(importance_rows, len(va)), replace=False)
            pi = permutation_importance(
                clf_full, Mva_full.iloc[sub], yva[sub], scoring="roc_auc",
                n_repeats=importance_repeats, random_state=seed, n_jobs=-1,
            )
            importance = (
                pd.DataFrame({"feature": full.columns, "importance_mean": pi.importances_mean, "importance_std": pi.importances_std})
                .sort_values("importance_mean", ascending=False)
                .reset_index(drop=True)
            )

    pooled = pd.DataFrame([{"model": n, **_binary_metrics(y, oof[n])} for n in names]).set_index("model")
    per_fold = pd.DataFrame(fold_rows)
    fold_summary = per_fold.groupby("model")[["roc_auc", "average_precision"]].agg(["mean", "std"])

    curves = {}
    for n in names:
        fpr, tpr, _ = roc_curve(y, oof[n])
        prec, rec, _ = precision_recall_curve(y, oof[n])
        curves[n] = {"fpr": fpr, "tpr": tpr, "precision": prec, "recall": rec}

    return {
        "y": y,
        "oof": oof,
        "metrics": pooled,
        "fold_metrics": per_fold,
        "fold_summary": fold_summary,
        "importance": importance,
        "curves": curves,
        "base_rate": float(y.mean()),
        "n_rows": int(len(y)),
        "id_lift_auc": float(pooled.loc["hgb_full", "roc_auc"] - pooled.loc["hgb_no_id", "roc_auc"]),
        "id_lift_ap": float(pooled.loc["hgb_full", "average_precision"] - pooled.loc["hgb_no_id", "average_precision"]),
        "runtime_s": time.time() - t0,
    }


# ----------------------------------------------------------------------------- experiment 2: quantile regression
def hierarchical_predictions(model: bl.HierarchicalQuantileModel, frame: pd.DataFrame, q: float) -> np.ndarray:
    """Evaluate the lookup model once per unique key tuple, then broadcast.

    `bid_for_winrate` recomputes quantiles on every call; grouping identical key
    tuples first keeps a 25k-row test set to a few thousand evaluations.
    """
    key_cols = [model.strata_col] + model.levels
    keys = frame[key_cols].astype(str).reset_index(drop=True)
    uniq = keys.drop_duplicates().reset_index(drop=True)
    uniq["_pred"] = uniq.apply(lambda r: model.bid_for_winrate(r, q), axis=1)
    return keys.merge(uniq, on=key_cols, how="left")["_pred"].to_numpy()


def run_quantile_experiment(
    bidded: pd.DataFrame,
    quantiles: tuple[float, ...] = (0.2, 0.5, 0.8),
    test_size: float = 0.2,
    seed: int = 0,
    m: float = 20.0,
) -> dict:
    """Clearing-price quantile regression versus the hierarchical lookup on unseen slots.

    The split is grouped by (url, ad_slot), so every test row belongs to a pair the
    models never saw during training. Target encodings are stratified by
    auction_type and computed out-of-fold on the training rows.
    """
    t0 = time.time()
    data = bidded[bidded["clearing_price"].notna()].copy()
    data["domain"] = data["domain"].where(data["domain"].notna(), NA_TOKEN)
    X = build_feature_frame(data)
    price = data["clearing_price"].to_numpy(dtype=float)
    y_log = np.log1p(price)
    groups = (data["url"].astype(str) + "|" + data["ad_slot"].astype(str)).to_numpy()

    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(X, y_log, groups))
    Xtr, Xte = X.iloc[tr], X.iloc[te]

    encoder = TargetEncoder(QR_ID_COLUMNS, m=m, strata_col="auction_type", seed=seed)
    te_tr = encoder.fit_transform_oof(Xtr, y_log[tr])
    te_te = encoder.transform(Xte)
    te_cols = [f"te_{c}" for c in QR_ID_COLUMNS]

    mat = BoosterMatrix(CATEGORICAL, QR_NUMERIC, te_cols).fit(Xtr)
    Mtr, Mte = mat.transform(Xtr, te_tr), mat.transform(Xte, te_te)

    # The lookup model sees exactly the same training rows.
    hier = bl.HierarchicalQuantileModel(levels=["url", "ad_slot", "domain"], pseudo_count=30).fit(data.iloc[tr])

    rows, preds = [], {"quantile_regression": {}, "hierarchical_lookup": {}}
    price_te = price[te]
    for q in quantiles:
        reg = HistGradientBoostingRegressor(
            loss="quantile", quantile=q, categorical_features=mat.categorical_mask,
            learning_rate=0.06, max_iter=400, max_leaf_nodes=31,
            min_samples_leaf=40, l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=30, random_state=seed,
        ).fit(Mtr, y_log[tr])
        p_qr = np.expm1(reg.predict(Mte))
        p_hier = hierarchical_predictions(hier, data.iloc[te], q)
        preds["quantile_regression"][q] = p_qr
        preds["hierarchical_lookup"][q] = p_hier
        for name, p in (("quantile_regression", p_qr), ("hierarchical_lookup", p_hier)):
            rows.append(
                {
                    "model": name,
                    "quantile": q,
                    "coverage": float(np.mean(price_te < p)),
                    "pinball_cpm": float(mean_pinball_loss(price_te, p, alpha=q)),
                    "pinball_log": float(mean_pinball_loss(np.log1p(price_te), np.log1p(np.clip(p, 0, None)), alpha=q)),
                    "median_bid_cpm": float(np.median(p)),
                }
            )
    table = pd.DataFrame(rows)
    table["coverage_gap"] = table["coverage"] - table["quantile"]

    # How cold is the cold start? Share of test rows whose keys exist in training.
    seen = {c: float(Xte[c].isin(set(Xtr[c])).mean()) for c in ["url", "ad_slot", "domain"]}

    # Coverage by auction type for the median, a check that neither regime is ignored.
    by_type = []
    for name in preds:
        p = preds[name][0.5]
        for at in ("FIRST_PRICE", "SECOND_PRICE"):
            mask = (Xte["auction_type"] == at).to_numpy()
            if mask.any():
                by_type.append({"model": name, "auction_type": at, "n": int(mask.sum()), "coverage_at_50": float(np.mean(price_te[mask] < p[mask]))})

    return {
        "table": table,
        "by_auction_type": pd.DataFrame(by_type),
        "preds": preds,
        "price_test": price_te,
        "n_train_rows": int(len(tr)),
        "n_test_rows": int(len(te)),
        "n_train_pairs": int(pd.Series(groups[tr]).nunique()),
        "n_test_pairs": int(pd.Series(groups[te]).nunique()),
        "test_key_seen_in_train": seen,
        "runtime_s": time.time() - t0,
    }


# ----------------------------------------------------------------------------- figures
_LABELS = {"hgb_full": "Boosting with ID encodings", "hgb_no_id": "Boosting, context only", "logreg": "Logistic baseline"}
_COLORS = {"hgb_full": vs.CATEGORICAL[0], "hgb_no_id": vs.CATEGORICAL[1], "logreg": vs.CATEGORICAL[2]}


def plot_ctr_roc_pr(res: dict):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for n in ["hgb_full", "hgb_no_id", "logreg"]:
        c = res["curves"][n]
        auc = res["metrics"].loc[n, "roc_auc"]
        ap = res["metrics"].loc[n, "average_precision"]
        axes[0].plot(c["fpr"], c["tpr"], lw=2, color=_COLORS[n], label=f"{_LABELS[n]} (AUC {auc:.3f})")
        axes[1].plot(c["recall"], c["precision"], lw=2, color=_COLORS[n], label=f"{_LABELS[n]} (AP {ap:.3f})")
    axes[0].plot([0, 1], [0, 1], color=vs.INK_MUTED, lw=1, ls="--")
    axes[1].axhline(res["base_rate"], color=vs.INK_MUTED, lw=1, ls="--")
    axes[1].text(0.99, res["base_rate"] + 0.01, f"base rate {res['base_rate']:.1%}", ha="right", fontsize=8, color=vs.INK_SECONDARY)
    vs.despine_labels(axes[0], "ROC, out-of-fold predictions", "false positive rate", "true positive rate")
    vs.despine_labels(axes[1], "Precision and recall", "recall", "precision")
    for ax in axes:
        ax.legend(loc="lower right" if ax is axes[0] else "upper right")
        ax.grid(True)
    return fig


def plot_ctr_calibration(res: dict, model: str = "hgb_full", n_bins: int = 10):
    import matplotlib.pyplot as plt

    y, p = res["y"], res["oof"][model]
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    top = max(mean_pred.max(), frac_pos.max()) * 1.05
    axes[0].plot([0, top], [0, top], color=vs.INK_MUTED, lw=1, ls="--")
    axes[0].plot(mean_pred, frac_pos, marker="o", ms=6, lw=2, color=_COLORS[model])
    for xp, yp in zip(mean_pred, frac_pos):
        axes[0].annotate(f"{yp:.2f}", (xp, yp), textcoords="offset points", xytext=(6, -10), fontsize=7, color=vs.INK_SECONDARY)
    vs.despine_labels(axes[0], "Reliability: predicted vs observed click rate", "mean predicted probability (decile bins)", "observed click rate")
    axes[0].grid(True)
    axes[1].hist(p, bins=40, color=_COLORS[model])
    axes[1].axvline(res["base_rate"], color=vs.STATUS["critical"], lw=1.2, ls="--")
    axes[1].text(res["base_rate"] + 0.005, axes[1].get_ylim()[1] * 0.9, f"base rate {res['base_rate']:.1%}", fontsize=8, color=vs.STATUS["critical"])
    vs.despine_labels(axes[1], "Distribution of predicted probabilities", "predicted probability of a click", "impressions")
    axes[1].grid(False)
    return fig


def plot_ctr_importance(res: dict, top: int = 15):
    import matplotlib.pyplot as plt

    imp = res["importance"].head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(imp) + 1.2))
    ax.hlines(imp["feature"], 0, imp["importance_mean"], color=vs.CATEGORICAL[0], lw=2)
    ax.plot(imp["importance_mean"], imp["feature"], "o", ms=7, color=vs.CATEGORICAL[0])
    ax.errorbar(imp["importance_mean"], imp["feature"], xerr=imp["importance_std"], fmt="none", ecolor=vs.INK_MUTED, elinewidth=1, capsize=2)
    for y_pos, v in enumerate(imp["importance_mean"]):
        ax.text(v + imp["importance_std"].max() * 0.3 + 0.0005, y_pos, f"{v:.4f}", va="center", fontsize=8, color=vs.INK_SECONDARY)
    vs.despine_labels(ax, "Permutation importance (drop in AUC when shuffled)", "mean AUC drop over 5 shuffles", None)
    ax.grid(False)
    ax.set_xlim(left=min(0, imp["importance_mean"].min() * 1.2))
    return fig


def plot_qr_coverage(res: dict):
    import matplotlib.pyplot as plt

    t = res["table"]
    fig, ax = plt.subplots(figsize=(6.4, 5))
    ax.plot([0, 1], [0, 1], color=vs.INK_MUTED, lw=1, ls="--")
    for name, color, marker in (("quantile_regression", vs.CATEGORICAL[0], "o"), ("hierarchical_lookup", vs.CATEGORICAL[1], "s")):
        sub = t[t["model"] == name]
        ax.plot(sub["quantile"], sub["coverage"], marker=marker, ms=8, lw=2, color=color,
                label="Quantile regression" if name == "quantile_regression" else "Hierarchical lookup")
        for xq, yc in zip(sub["quantile"], sub["coverage"]):
            ax.annotate(f"{yc:.2f}", (xq, yc), textcoords="offset points",
                        xytext=(8, -4) if name == "quantile_regression" else (-30, 6), fontsize=8, color=color)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    vs.despine_labels(ax, "Unseen slots: target quantile vs realised win rate", "target quantile (intended win rate)", "share of test clearing prices below the bid")
    ax.legend(loc="upper left")
    ax.grid(True)
    return fig


def plot_qr_pinball(res: dict):
    import matplotlib.pyplot as plt

    t = res["table"]
    qs = sorted(t["quantile"].unique())
    width = 0.36
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, (name, color) in enumerate((("quantile_regression", vs.CATEGORICAL[0]), ("hierarchical_lookup", vs.CATEGORICAL[1]))):
        sub = t[t["model"] == name].set_index("quantile").loc[qs]
        xs = np.arange(len(qs)) + (i - 0.5) * width
        ax.bar(xs, sub["pinball_cpm"], width=width, color=color,
               label="Quantile regression" if name == "quantile_regression" else "Hierarchical lookup")
        for x, v in zip(xs, sub["pinball_cpm"]):
            ax.text(x, v + t["pinball_cpm"].max() * 0.015, f"{v:.3f}", ha="center", fontsize=8, color=vs.INK_SECONDARY)
    ax.set_xticks(np.arange(len(qs)))
    ax.set_xticklabels([f"q = {q:.1f}" for q in qs])
    vs.despine_labels(ax, "Pinball loss on unseen slots (lower is better)", None, "pinball loss, CPM")
    ax.legend()
    ax.grid(False)
    return fig
