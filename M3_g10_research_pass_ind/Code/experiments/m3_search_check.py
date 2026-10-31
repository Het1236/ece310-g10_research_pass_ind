"""M3 check 1: do we need a certified search for the antenna position?

EDMA places the antenna with golden-section search. That search is exact only if the objective
has one peak along the waveguide. I test this for both design problems of the final method:

  P1  maximise the sum rate of one segment (all users at the power cap).
  P2  maximise the minimum user rate (max-min fairness), with the best powers at every position.

For K = 2 and K = 4 users in random positions I count the peaks of the objective and I compare
  - plain golden-section search (EDMA, one bracket),
  - the certified global search of the final method (Piyavskii-Shubert with a proven Lipschitz bound),
against a dense grid search used as the reference (4001 points per drop).
The interference floor moves with the antenna position in every case.

Run from the Code folder:
    python experiments/m3_search_check.py            # 200 drops per K and problem
    python experiments/m3_search_check.py --quick    # 40 drops per K and problem
"""
from __future__ import annotations

import argparse
import csv

import numpy as np

from m3_common import RESULTS_DIR, MAROON, GREY, BLUE, GREEN, save_figure, setup_matplotlib

from src import floor_position, maxmin_global, placement
from src.config import load_params

K_LIST = [2, 4]
GRID = 4001


def count_peaks(f: np.ndarray, rel_tol: float = 1e-9) -> int:
    """Number of strict local maxima of a sampled curve, ends included."""
    df = np.diff(f)
    scale = float(np.max(np.abs(f))) * rel_tol
    sign = np.where(df > scale, 1, np.where(df < -scale, -1, 0))
    sign = sign[sign != 0]
    if sign.size == 0:
        return 1
    peaks = int(np.sum((sign[:-1] > 0) & (sign[1:] < 0)))
    peaks += int(sign[0] < 0) + int(sign[-1] > 0)
    return max(peaks, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    n_drops = 40 if args.quick else 200

    p = load_params()
    D_S, D_W, d, eta = float(p["D_S"]), float(p["D_W"]), float(p["d"]), float(p["eta"])
    P = float(p["phase4"]["P_max"])
    tol = float(p["phase4"]["gs_tol"])
    B_F = floor_position.floor_logderiv_bound(p)
    rng = np.random.default_rng(int(p["seed"]))
    xs = np.linspace(0.0, D_S, GRID)

    rows, worst = [], {}
    for problem in ("P1", "P2"):
        for K in K_LIST:
            Ffn = floor_position.floor_fn_absolute(K, p, 0.0)
            F_arr = np.array([Ffn(x) for x in xs])
            multi = miss_gs = miss_cert = 0
            def_gs, def_cert, gaps, evals, evals_gs = [], [], [], [], []
            for _ in range(n_drops):
                ux = rng.uniform(0.0, D_S, K)
                uy = rng.uniform(-D_W / 2.0, D_W / 2.0, K)
                if problem == "P1":
                    f_scalar = lambda x, ux=ux, uy=uy: placement.rate_at_position(x, ux, uy, eta, d, P, Ffn)
                    G = placement.collected_power(xs, ux, uy, eta, d, P)
                    f_grid = np.log2(1.0 + G / F_arr)
                    cert = maxmin_global.deterministic_global
                else:
                    f_scalar = maxmin_global.maxmin_value_function(ux, uy, d, eta, Ffn, "A", P_max=P)
                    f_grid = np.array([f_scalar(float(x)) for x in xs])
                peaks = count_peaks(f_grid)
                multi += int(peaks > 1)
                f_ref = float(np.max(f_grid))
                _, f_gs, n_gs = placement.golden_section(f_scalar, 0.0, D_S, tol)
                evals_gs.append(n_gs + 3)
                if problem == "P2":
                    res = maxmin_global.deterministic_global(f_scalar, ux, uy, d, 0.0, D_S, tol=1e-3, floor_logderiv=B_F)
                else:
                    # P1 has no closed-form value function here, so the certified search runs on the rate itself
                    # with the same Lipschitz constant (Theorem 3 (ii) covers the P1 objective).
                    res = maxmin_global.deterministic_global(f_scalar, ux, uy, d, 0.0, D_S, tol=1e-3, floor_logderiv=B_F)
                dg = max(0.0, 100.0 * (f_ref - f_gs) / f_ref)
                dc = max(0.0, 100.0 * (f_ref - res["f"]) / f_ref)
                def_gs.append(dg)
                def_cert.append(dc)
                gaps.append(res["gap"])
                evals.append(res["n_eval"])
                miss_gs += int(dg > 1e-3)
                miss_cert += int(dc > 1e-3)
                if problem not in worst or dg > worst[problem]["deficit"]:
                    worst[problem] = {"K": K, "ux": ux.copy(), "uy": uy.copy(), "deficit": dg,
                                      "gs": placement.golden_section(f_scalar, 0.0, D_S, tol)[:2],
                                      "cert": (res["x"], res["f"]), "curve": f_grid.copy()}
            row = {"problem": problem, "K": K, "drops": n_drops,
                   "multi_peak_pct": 100.0 * multi / n_drops,
                   "plain_gs_miss_pct": 100.0 * miss_gs / n_drops,
                   "plain_gs_worst_deficit_pct": float(np.max(def_gs)),
                   "certified_miss_pct": 100.0 * miss_cert / n_drops,
                   "certified_worst_deficit_pct": float(np.max(def_cert)),
                   "certified_max_gap": float(np.max(gaps)),
                   "certified_mean_evals": float(np.mean(evals)),
                   "golden_section_evals": float(np.mean(evals_gs))}
            rows.append(row)
            print(f"{problem} K={K}: several peaks in {row['multi_peak_pct']:5.1f} % of drops; plain golden-section "
                  f"misses {row['plain_gs_miss_pct']:5.1f} % (worst {row['plain_gs_worst_deficit_pct']:.2f} %); certified misses "
                  f"{row['certified_miss_pct']:.1f} % (worst {row['certified_worst_deficit_pct']:.1e} %, largest certified gap "
                  f"{row['certified_max_gap']:.1e}, mean {row['certified_mean_evals']:.0f} evaluations)", flush=True)

    path = RESULTS_DIR / "search_check.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {path}")

    plt = setup_matplotlib()
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.0))
    w = worst["P2"]
    ax[0].plot(xs, w["curve"], color=MAROON, lw=2, label=f"max-min floor, K = {w['K']}")
    ax[0].plot([w["gs"][0]], [w["gs"][1]], "s", color=BLUE, ms=9, label=f"plain golden-section ({w['deficit']:.1f} % below best)")
    ax[0].plot([w["cert"][0]], [w["cert"][1]], "*", color=GREEN, ms=14, label="certified global search")
    for u in w["ux"]:
        ax[0].axvline(u, color=GREY, lw=0.8, ls=":")
    ax[0].set_xlabel("antenna position in the segment [m]")
    ax[0].set_ylabel("max-min floor [bps/Hz]")
    ax[0].set_title("Worst random drop found for P2 (dotted: users)", fontsize=11)
    ax[0].legend(fontsize=8.5, loc="lower center")

    labels = [f"{r['problem']}, K = {r['K']}" for r in rows]
    xk = np.arange(len(rows))
    wd = 0.27
    ax[1].bar(xk - wd, [r["multi_peak_pct"] for r in rows], wd, color=GREY, label="objective has several peaks")
    ax[1].bar(xk, [r["plain_gs_miss_pct"] for r in rows], wd, color=BLUE, label="plain golden-section misses")
    ax[1].bar(xk + wd, [r["certified_miss_pct"] for r in rows], wd, color=GREEN, label="certified search misses")
    ax[1].set_xticks(xk)
    ax[1].set_xticklabels(labels, fontsize=9)
    ax[1].set_ylabel("share of random drops [%]")
    ax[1].legend(fontsize=9)
    fig.tight_layout()
    save_figure(fig, "search_check.png")


if __name__ == "__main__":
    main()
