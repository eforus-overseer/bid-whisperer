"""Run both machine-learning experiments end to end.

Writes reports/metrics_models.json and five figures under reports/figures/.
Usage:  python3 scripts/run_models.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import warnings  # noqa: E402

warnings.filterwarnings("ignore")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import data_loading as dl  # noqa: E402
import models as ml  # noqa: E402
import viz_style as vs  # noqa: E402

FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def main() -> None:
    t0 = time.time()
    vs.apply_style()
    df = dl.add_derived(dl.load_raw())
    won, bidded = dl.won_frame(df), dl.bidded_frame(df)

    print("CTR experiment ...")
    ctr = ml.run_ctr_experiment(won)
    print(ctr["metrics"].round(4).to_string())
    print(f"  AUC lift from ID encodings: {ctr['id_lift_auc']:+.4f} | AP lift: {ctr['id_lift_ap']:+.4f} | {ctr['runtime_s']:.0f}s")
    print(ctr["importance"].head(8).round(4).to_string(index=False))

    print("\nQuantile regression experiment ...")
    qr = ml.run_quantile_experiment(bidded)
    print(qr["table"].round(4).to_string(index=False))
    print(qr["by_auction_type"].round(4).to_string(index=False))
    print(f"  train pairs {qr['n_train_pairs']:,} / test pairs {qr['n_test_pairs']:,} | test keys seen in train: {qr['test_key_seen_in_train']} | {qr['runtime_s']:.0f}s")

    for name, fig in (
        ("ml_ctr_roc_pr.png", ml.plot_ctr_roc_pr(ctr)),
        ("ml_ctr_calibration.png", ml.plot_ctr_calibration(ctr)),
        ("ml_ctr_importance.png", ml.plot_ctr_importance(ctr)),
        ("ml_qr_coverage.png", ml.plot_qr_coverage(qr)),
        ("ml_qr_pinball.png", ml.plot_qr_pinball(qr)),
    ):
        fig.savefig(FIG / name, dpi=130, bbox_inches="tight")

    out = {
        "ctr": {
            "n_rows": ctr["n_rows"],
            "base_rate": ctr["base_rate"],
            "pooled_oof_metrics": ctr["metrics"].round(5).to_dict(orient="index"),
            "fold_summary": {m: {f"{a}_{b}": float(v) for (a, b), v in row.items()} for m, row in ctr["fold_summary"].round(5).to_dict(orient="index").items()},
            "id_encoding_lift": {"roc_auc": ctr["id_lift_auc"], "average_precision": ctr["id_lift_ap"]},
            "permutation_importance": ctr["importance"].round(5).to_dict(orient="records"),
            "runtime_s": round(ctr["runtime_s"], 1),
        },
        "quantile_regression": {
            "split": {
                "n_train_rows": qr["n_train_rows"], "n_test_rows": qr["n_test_rows"],
                "n_train_pairs": qr["n_train_pairs"], "n_test_pairs": qr["n_test_pairs"],
                "test_key_seen_in_train": qr["test_key_seen_in_train"],
            },
            "table": qr["table"].round(5).to_dict(orient="records"),
            "by_auction_type_at_q50": qr["by_auction_type"].round(5).to_dict(orient="records"),
            "runtime_s": round(qr["runtime_s"], 1),
        },
        "total_runtime_s": round(time.time() - t0, 1),
    }
    (ROOT / "reports" / "metrics_models.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote reports/metrics_models.json and 5 figures in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
