"""M2 baseline reproduction: EDMA Table I (one user per segment, K = 1).

I reproduce the published numbers of Ding, Schober and Poor, "Environment Division
Multiple Access (EDMA): A Feasibility Study via Pinching Antennas", IEEE TWC 2026, Table I.
This checks that my channel model, my simulator and my closed-form code agree with the
paper I am building on before I add more users per segment.

Setup (same as EDMA): 28 GHz, D_L = 4 D_W = 40 m, D_W = 10 m, d = 3 m, phi = 0.02,
transmit SNR 120 dB. The EDMA scheme is simulated by Monte Carlo with full interference.
Time division (TDMA) is computed in closed form for two antenna placements.

Run from the Code folder:
    python experiments/m2_baseline_edma.py            # 150000 trials
    python experiments/m2_baseline_edma.py --quick    # 20000 trials
"""
from __future__ import annotations

import argparse
import copy
import csv

from m2_common import RESULTS_DIR, MAROON, BLUE, GREY, save_figure, setup_matplotlib

from src import analytic, montecarlo
from src.config import load_params

# Published sum rates [bps/Hz] from EDMA Table I, K = 1.
PUBLISHED = {
    "edma":  {3: 41.07, 5: 51.80, 7: 50.28, 9: 43.91, 11: 37.15},
    "ptdma": {3: 5.69, 5: 3.56, 7: 2.61, 9: 2.07, 11: 1.72},
    "ctdma": {3: 5.41, 5: 3.48, 7: 2.57, 9: 2.05, 11: 1.72},
}
M_LIST = [3, 5, 7, 9, 11]
SEED = 2


def run(n_trials: int) -> list[dict]:
    base = load_params()
    base["D_L"] = 4.0 * float(base["D_W"])
    rows = []
    for M in M_LIST:
        q = copy.deepcopy(base)
        q["M"] = M
        q["D_S"] = q["D_L"] / M
        q["delta"] = q["D_S"] / 2.0
        edma = montecarlo.system_sum_rate_mc_triangular(M, q, n_trials=n_trials, seed=SEED)
        ptdma = analytic.tdma_rate_eq9("B", 1, float(M), q, placement="u0")
        ctdma = analytic.tdma_rate_eq9("B", 1, float(M), q, placement="uavg")
        rows.append({
            "M": M,
            "edma_published": PUBLISHED["edma"][M],
            "edma_ours": edma["mean_rate"],
            "edma_stderr": edma["stderr"],
            "edma_err_pct": 100 * abs(edma["mean_rate"] - PUBLISHED["edma"][M]) / PUBLISHED["edma"][M],
            "ptdma_published": PUBLISHED["ptdma"][M],
            "ptdma_ours": ptdma,
            "ptdma_err_pct": 100 * abs(ptdma - PUBLISHED["ptdma"][M]) / PUBLISHED["ptdma"][M],
            "ctdma_published": PUBLISHED["ctdma"][M],
            "ctdma_ours": ctdma,
            "ctdma_err_pct": 100 * abs(ctdma - PUBLISHED["ctdma"][M]) / PUBLISHED["ctdma"][M],
        })
    return rows


def make_figure(rows: list[dict]) -> None:
    plt = setup_matplotlib()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    Ms = [r["M"] for r in rows]

    ax[0].plot(Ms, [r["edma_published"] for r in rows], "o-", color=GREY, label="EDMA paper, Table I")
    ax[0].errorbar(Ms, [r["edma_ours"] for r in rows], yerr=[3 * r["edma_stderr"] for r in rows],
                   fmt="s--", color=MAROON, label="my simulation (3 sigma bars)")
    ax[0].set_xlabel("number of segments M")
    ax[0].set_ylabel("system sum rate [bps/Hz]")
    ax[0].set_title("EDMA scheme, K = 1")
    ax[0].legend()

    ax[1].plot(Ms, [r["ptdma_published"] for r in rows], "o-", color=GREY, label="paper, antenna follows user")
    ax[1].plot(Ms, [r["ptdma_ours"] for r in rows], "s--", color=MAROON, label="mine, antenna follows user")
    ax[1].plot(Ms, [r["ctdma_published"] for r in rows], "^-", color="#9CA3AF", label="paper, antenna at centre")
    ax[1].plot(Ms, [r["ctdma_ours"] for r in rows], "v--", color=BLUE, label="mine, antenna at centre")
    ax[1].set_xlabel("number of segments M")
    ax[1].set_title("time division (TDMA)")
    ax[1].legend(fontsize=9)
    fig.tight_layout()
    save_figure(fig, "baseline_edma_table1.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    n = 20_000 if args.quick else 150_000

    rows = run(n)
    print(f"{'M':>3} {'EDMA paper':>10} {'mine':>8} {'err %':>6} | {'P-TDMA paper':>12} {'mine':>7} | "
          f"{'C-TDMA paper':>12} {'mine':>7}")
    for r in rows:
        print(f"{r['M']:>3} {r['edma_published']:>10.2f} {r['edma_ours']:>8.2f} {r['edma_err_pct']:>6.2f} | "
              f"{r['ptdma_published']:>12.2f} {r['ptdma_ours']:>7.2f} | "
              f"{r['ctdma_published']:>12.2f} {r['ctdma_ours']:>7.2f}")
    worst = max(max(r["edma_err_pct"], r["ptdma_err_pct"], r["ctdma_err_pct"]) for r in rows)
    print(f"largest relative difference to the paper: {worst:.2f} %   (trials = {n}, seed = {SEED})")

    path = RESULTS_DIR / "baseline_edma_table1.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {path}")
    make_figure(rows)


if __name__ == "__main__":
    main()
