"""M2 feasibility check: does the far-segment leakage stay negligible when K users share a segment?

Question. EDMA ignores segments that are not direct neighbours because blockage kills their
leakage. It proves this for one user per segment. With K users per segment every segment leaks
K times more. If the far and adjacent leakage both scale with K, the ratio far/adjacent should
not depend on K.

What this script does.
  1. Monte Carlo of the true far/adjacent leakage ratio for K = 1, 5, 20, 50.
  2. The closed-form bound  e^{-x} / (1 - e^{-x})  with x = phi * D_S^2.
  3. My closed-form estimate (sum of per-segment weights) as a dashed line.

Run from the Code folder:
    python experiments/m2_leakage_ratio.py            # N = 50000, tail N = 400000
    python experiments/m2_leakage_ratio.py --quick    # N = 3000
"""
from __future__ import annotations

import argparse
import csv
import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from m2_common import RESULTS_DIR, MAROON, GREY, BLUE, GREEN, ORANGE, save_figure, setup_matplotlib

from src import analytic, montecarlo
from src.config import load_params

K_LIST = [1, 5, 20, 50]
X_POINTS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]   # x = phi * D_S^2
RESOLVABLE_SIGMA = 3.0   # a point is drawn only if ratio > 3 * standard error


def D_S_for(x: float, phi: float) -> float:
    return math.sqrt(x / phi)


def bound(x: float) -> float:
    return math.exp(-x) / (1.0 - math.exp(-x))


def run_point(task: tuple) -> dict:
    """One (K, x) point: Monte Carlo ratio plus the closed-form numbers."""
    K, x, n, params, seed, ell_max, phi = task
    p = dict(params)
    p["D_S"] = D_S_for(x, phi)
    res = montecarlo.ratio_mc(p, p["D_S"] / 2.0, n_trials=n, seed=seed, users_per_segment=K)
    est = analytic.ratio_R(p, delta=p["D_S"] / 2.0, ell_max=ell_max)
    return {"K": K, "x_phiDs2": x, "n_trials": n, "ratio_mc": res["ratio"],
            "stderr": res["stderr"], "bound": bound(x), "estimate": est}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    n_low, n_tail = (3000, 3000) if args.quick else (50_000, 400_000)

    params = load_params()
    phi = float(params["phi"])
    seed = int(params["seed"])
    ell_max = int(params["phase2_far_probe_segments"]) // 2

    # Every (K, x) point is independent and has its own fixed seed, so the points
    # can run in parallel and the numbers stay the same.
    tasks = [(K, x, n_tail if x >= 2.5 else n_low, params, seed, ell_max, phi)
             for K in K_LIST for x in X_POINTS]
    rows = []
    with ProcessPoolExecutor() as pool:
        for row in pool.map(run_point, tasks):
            rows.append(row)
            print(f"K={row['K']:<3d} x={row['x_phiDs2']:<4.1f} N={row['n_trials']:<7d} "
                  f"mc={row['ratio_mc']:.3e} +- {row['stderr']:.1e}   "
                  f"bound={row['bound']:.3e}   estimate={row['estimate']:.3e}", flush=True)

    path = RESULTS_DIR / "leakage_ratio.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {path}")

    # Checks that I report in the M2 report.
    ok = [r for r in rows if r["ratio_mc"] > RESOLVABLE_SIGMA * r["stderr"]]
    worst = max(r["ratio_mc"] / r["bound"] for r in ok)
    print(f"resolvable points: {len(ok)} of {len(rows)}; largest ratio_mc / bound = {worst:.3f} (must be < 1)")
    for x in (1.0, 2.0, 3.0):
        vals = [r["ratio_mc"] for r in rows if r["x_phiDs2"] == x and r["ratio_mc"] > RESOLVABLE_SIGMA * r["stderr"]]
        if len(vals) > 1:
            print(f"x = {x}: ratio over K = {min(vals):.3e} .. {max(vals):.3e}  (spread {100*(max(vals)/min(vals)-1):.1f} %)")

    x_line = np.linspace(0.5, 5.0, 200)
    est_line = []
    for x in x_line:
        p = dict(params)
        p["D_S"] = D_S_for(x, phi)
        est_line.append(analytic.ratio_R(p, delta=p["D_S"] / 2.0, ell_max=ell_max))

    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.semilogy(x_line, [bound(x) for x in x_line], "-", color=MAROON, lw=2, label=r"bound $e^{-x}/(1-e^{-x})$")
    ax.semilogy(x_line, est_line, "--", color=GREY, lw=1.6, label="my closed-form estimate")
    colors = {1: BLUE, 5: GREEN, 20: ORANGE, 50: "#7B3F98"}
    for K in K_LIST:
        pts = [r for r in rows if r["K"] == K and r["ratio_mc"] > RESOLVABLE_SIGMA * r["stderr"]]
        ax.errorbar([r["x_phiDs2"] for r in pts], [r["ratio_mc"] for r in pts],
                    yerr=[RESOLVABLE_SIGMA * r["stderr"] for r in pts], fmt="o", ms=5,
                    color=colors[K], label=f"Monte Carlo, K = {K}")
    ax.axhline(0.05, color="black", lw=0.8, ls=":")
    ax.text(4.95, 0.056, "5 % target", ha="right", fontsize=9)
    ax.axvline(math.log(21), color="black", lw=0.8, ls=":")
    ax.text(math.log(21) + 0.05, 2e-6, r"gate $x=\ln 21$", fontsize=9)
    ax.set_xlabel(r"$x=\varphi D_S^2$")
    ax.set_ylabel("far / adjacent leakage ratio")
    ax.set_ylim(1e-6, 2)
    ax.legend(fontsize=9, loc="lower left")
    save_figure(fig, "leakage_ratio_vs_phiDs2.png")


if __name__ == "__main__":
    main()
