"""Run the extra analysis layers and the obfuscation forensics end to end.

Writes every layer figure to reports/figures/ and the headline numbers to
reports/metrics_layers.json. Usage: python3 scripts/run_layers.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

import data_loading as dl  # noqa: E402
import decode as dc  # noqa: E402
import layer_plots as lp  # noqa: E402
import layers as ly  # noqa: E402
import viz_style as vs  # noqa: E402

FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def save(fig, name: str) -> None:
    fig.savefig(FIG / name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


def main() -> None:
    t0 = time.time()
    vs.apply_style()
    df = dl.add_derived(dl.load_raw())
    bidded, won = dl.bidded_frame(df), dl.won_frame(df)
    overall_ctr = float(won["conversion"].mean())
    metrics: dict = {"users": ly.user_uniqueness_report(df)}

    print("1. bid shading")
    stats = ly.overpay_stats(bidded)
    curve = ly.shading_curve(bidded)
    oracle = ly.oracle_curve(bidded)
    metrics["overpay"] = stats
    metrics["shading_curve"] = curve.to_dict(orient="records")
    metrics["oracle_curve"] = oracle.to_dict(orient="records")
    save(lp.plot_overpay(ly.overpay_frame(bidded), curve, stats), "layer_overpay.png")

    print("2. hourly")
    hp = ly.hourly_profile(df, bidded, won)
    metrics["hourly"] = hp.to_dict(orient="records")
    save(lp.plot_hourly(hp), "layer_hourly.png")

    print("3. concentration")
    conc = ly.concentration_table(df)
    metrics["concentration"] = conc.to_dict(orient="records")
    curves = {c: (ly.lorenz_points(df[c].value_counts().to_numpy()), ly.gini(df[c].value_counts().to_numpy())) for c in ("domain", "url", "ad_slot")}
    save(lp.plot_lorenz(curves, n_domains=int(df["domain"].nunique())), "layer_lorenz.png")

    print("4. browser and OS")
    browser, os_table = ly.browser_os_summary(df, won)
    metrics["browser"] = browser.to_dict(orient="records")
    metrics["os"] = os_table.to_dict(orient="records")
    save(lp.plot_browser_os(browser, os_table, overall_ctr), "layer_browser_os.png")

    print("5. cookie age")
    ck = ly.cookie_age_ctr(won)
    metrics["cookie_age"] = ck.assign(cookie_age=ck["cookie_age"].astype(str)).to_dict(orient="records")
    save(lp.plot_cookie_age(ck, overall_ctr), "layer_cookie_age.png")

    print("6. publisher archetypes")
    fp = ly.domain_fingerprints(df, bidded, won, min_n=100)
    arch = ly.publisher_archetypes(fp)
    metrics["archetypes"] = {
        "n_domains": int(len(fp)),
        "best_k": arch["best_k"],
        "silhouette": arch["silhouette"].to_dict(orient="records"),
        "centroids": arch["centroids"].to_dict(orient="records"),
        "desktop_only_domains": int((fp["mobile_share"] < 0.05).sum()),
        "mobile_only_domains": int((fp["mobile_share"] > 0.95).sum()),
    }
    save(lp.plot_archetypes(arch["labelled"], arch["names"]), "layer_archetypes.png")

    print("7. states")
    states = ly.state_summary(df, bidded, won)
    unknown = sorted(set(states["state_code"].dropna()) - set(lp.STATE_GRID))
    if unknown:
        print(f"  state codes not on the tile grid: {unknown}")
    metrics["states"] = states.to_dict(orient="records")
    metrics["states_unknown_codes"] = unknown
    save(lp.plot_state_tile_map(states), "layer_state_map.png")

    print("8. obfuscation forensics")
    fmt = dc.hash_format_report(df)
    fd = dc.functional_dependency_matrix(df)
    nib = dc.nibble_frequencies(df)
    metrics["decode"] = {"format": fmt.to_dict(orient="records"), "functional_dependency": fd.round(4).to_dict(), "chained": dc.chained_hash_test(df).to_dict(orient="records")}
    save(lp.plot_fd_matrix(fd), "decode_fd_matrix.png")
    save(lp.plot_nibbles(nib), "decode_nibbles.png")

    sets = dc.hash_sets(df)
    ints = dc.integer_bruteforce(sets, n=1_000_000)
    metrics["decode"]["integer_bruteforce"] = {"n": ints["n"], "hits": len(ints["hits"]), "seconds": round(ints["seconds"], 1)}
    dictionary = dc.dictionary_attack(sets, limit=None)
    metrics["decode"]["dictionary"] = {k: (v if k != "hits" else len(v)) for k, v in dictionary.items()}
    print(f"  dictionary attack: {dictionary['status']}, digests={dictionary.get('digests', 0):,}, hits={len(dictionary['hits'])}")

    out = ROOT / "reports" / "metrics_layers.json"
    out.write_text(json.dumps(metrics, indent=2, default=lambda o: float(o) if hasattr(o, "__float__") else str(o)))
    print(f"wrote {out.relative_to(ROOT)} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
