"""M2 core result: exact ergodic uplink rate of one segment with K users, checked by Monte Carlo.

The closed integral for the ergodic rate lives in src/analytic.py (ergodic_rate_exact). This script
checks it against a direct Monte Carlo simulation of the same model and shows how the rate of one
segment changes with the load K.

Model used here: M = 3, 120 dB transmit SNR, K home users in LoS, 2K users in the two adjacent
segments that leak through the blockage law. Successive interference cancellation is assumed ideal.

Run from the Code folder:
    python experiments/m2_ergodic_rate.py            # N = 50000 per K
    python experiments/m2_ergodic_rate.py --quick    # N = 5000 per K
"""
from __future__ import annotations

import argparse
import csv

import numpy as np

from m2_common import RESULTS_DIR, MAROON, GREY, BLUE, save_figure, setup_matplotlib

from src import analytic, montecarlo
from src.config import load_params

K_CHECK = [1, 2, 5, 10, 20, 50]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    n = 5_000 if args.quick else 50_000

    params = load_params()
    seed = int(params["seed"])
    r1 = analytic.ergodic_rate_exact(1, params)[0]

    rows = []
    for K in K_CHECK:
        exact, trunc = analytic.ergodic_rate_exact(K, params)
        rng = np.random.default_rng(seed + K)
        mc = montecarlo.ergodic_rate_mc(K, params, "A", rng, n_trials=n)
        z = (mc["mean_rate"] - exact) / mc["stderr"]
        rows.append({"K": K, "analytic": exact, "trunc_error": trunc, "monte_carlo": mc["mean_rate"],
                     "stderr": mc["stderr"], "z_score": z, "ratio_to_K1": exact / r1,
                     "windfall_stripped": analytic.R_tilde(K, params)})
        print(f"K={K:<3d} analytic={exact:8.4f}  MC={mc['mean_rate']:8.4f} +- {mc['stderr']:.4f}"
              f"   z={z:+.2f}   R(K)/R(1)={exact/r1:.3f}", flush=True)

    path = RESULTS_DIR / "ergodic_rate_vs_K.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {path}")
    print(f"largest |z| = {max(abs(r['z_score']) for r in rows):.2f}  (trials per K = {n})")

    k_line = list(range(1, 65))
    line = [analytic.ergodic_rate_exact(k, params)[0] for k in k_line]
    line_tilde = [analytic.R_tilde(k, params) for k in k_line]

    plt = setup_matplotlib()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
    ax[0].plot(k_line, line, "-", color=MAROON, lw=2, label="closed integral")
    ax[0].plot(k_line, line_tilde, "--", color=GREY, lw=1.6, label="blockage windfall removed")
    ax[0].errorbar([r["K"] for r in rows], [r["monte_carlo"] for r in rows],
                   yerr=[3 * r["stderr"] for r in rows], fmt="o", color=BLUE, ms=5, label="Monte Carlo (3 sigma)")
    ax[0].set_xlabel("users per segment K")
    ax[0].set_ylabel("ergodic rate of one segment [bps/Hz]")
    ax[0].legend(fontsize=9)

    ax[1].plot(k_line, [v / line[0] for v in line], "-", color=MAROON, lw=2)
    ax[1].set_xlabel("users per segment K")
    ax[1].set_ylabel(r"$\bar R(K)\,/\,\bar R(1)$")
    ax[1].set_ylim(0, 1.05)
    fig.tight_layout()
    save_figure(fig, "ergodic_rate_vs_K.png")


if __name__ == "__main__":
    main()
