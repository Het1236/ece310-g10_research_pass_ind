"""M3 result 5: how many segments, and what does orthogonal access give at the same load?

Part 1. System sum rate of EPMA against the number of segments M, for several loads K and for
several service lengths D_L. The aim is to see where the best M sits. The exact edge-aware rate
is used, so far segments are included.

Part 2. At the operating point (M = 3, D_L = 40 m) I compare EPMA with system-wide orthogonal access
(users take turns, no interference) for K = 1 to 64 users per segment. The orthogonal system rate is
tdma_rate_eq9 times M K. Version B (average power) is the strong case for orthogonal access and
Version A (peak power) the weak one. The rates are system rates (M times the per-segment rate). EPMA uses the two-neighbour ergodic rate of the design model.

Run from the Code folder:
    python experiments/m3_segments.py            # full run
    python experiments/m3_segments.py --quick    # coarse grids
"""
from __future__ import annotations

import argparse
import copy
import csv

import numpy as np

from m3_common import RESULTS_DIR, MAROON, GREY, BLUE, GREEN, ORANGE, save_figure, setup_matplotlib

from src import analytic
from src.config import load_params


def variant(p, **over):
    """Copy of the parameters with the geometry re-derived (D_S = D_L / M, delta = D_S / 2)."""
    q = copy.deepcopy(p)
    for k, v in over.items():
        q[k] = int(v) if k == "M" else float(v)
    q["D_S"] = float(q["D_L"]) / float(q["M"])
    q["delta"] = float(q["D_S"]) / 2.0
    return q


def write_csv(name, rows):
    with open(RESULTS_DIR / name, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    p = load_params()

    M_grid = np.arange(2, 13, 2 if args.quick else 1)
    K_family = [1, 2, 4, 8]
    DL_family = [30.0, 40.0, 50.0, 60.0]
    rows = []
    best_K, best_DL = {}, {}
    for K in K_family:
        vals = [analytic.system_sum_rate_exact(K, int(M), variant(p, M=M))[0] for M in M_grid]
        for M, v in zip(M_grid, vals):
            rows.append({"study": "load", "K": K, "D_L": float(p["D_L"]), "M": int(M), "system_rate": float(v)})
        best_K[K] = int(M_grid[int(np.argmax(vals))])
        print(f"load K={K}: best M = {best_K[K]} (rate {max(vals):.2f})", flush=True)
    for D_L in DL_family:
        vals = [analytic.system_sum_rate_exact(1, int(M), variant(p, M=M, D_L=D_L))[0] for M in M_grid]
        for M, v in zip(M_grid, vals):
            rows.append({"study": "length", "K": 1, "D_L": D_L, "M": int(M), "system_rate": float(v)})
        best_DL[D_L] = int(M_grid[int(np.argmax(vals))])
        print(f"length D_L={D_L:.0f} m, K=1: best M = {best_DL[D_L]} (rate {max(vals):.2f})", flush=True)
    write_csv("segments.csv", rows)

    # Part 2: EPMA against orthogonal access at the operating point
    M0 = int(p["M"])
    K_grid = [1, 2, 4, 8, 16, 20, 32, 64] if not args.quick else [1, 4, 16, 64]
    cmp_rows = []
    for K in K_grid:
        epma = M0 * analytic.ergodic_rate_exact(K, p)[0]
        tb = analytic.tdma_system_rate("B", K, M0, p)
        ta = analytic.tdma_system_rate("A", K, M0, p)
        cmp_rows.append({"K": K, "epma": float(epma), "orthogonal_B": float(tb), "orthogonal_A": float(ta),
                         "ratio_B": float(epma / tb), "ratio_A": float(epma / ta),
                         "delay_epma": 1.0, "delay_orthogonal": float(M0 * K)})
        r = cmp_rows[-1]
        print(f"K={K:<3d}: EPMA {r['epma']:.2f}, orthogonal B {r['orthogonal_B']:.2f} (x{r['ratio_B']:.2f}), "
              f"orthogonal A {r['orthogonal_A']:.2f} (x{r['ratio_A']:.2f}); access delay 1 slot against {r['delay_orthogonal']:.0f}", flush=True)
    write_csv("epma_vs_orthogonal.csv", cmp_rows)
    cross = [r["K"] for r in cmp_rows if r["ratio_B"] < 1.0]
    print("first load where orthogonal access (average power) is ahead:", cross[0] if cross else "none in grid")

    plt = setup_matplotlib()
    fig, ax = plt.subplots(1, 3, figsize=(11.8, 3.9))
    cols = [MAROON, BLUE, GREEN, ORANGE]
    for K, c in zip(K_family, cols):
        r = [x for x in rows if x["study"] == "load" and x["K"] == K]
        ax[0].plot([x["M"] for x in r], [x["system_rate"] for x in r], "o-", color=c, lw=1.8, label=f"K = {K}")
    ax[0].set_xlabel("segments M")
    ax[0].set_ylabel("system sum rate [bps/Hz]")
    ax[0].set_title("A. Load changes the best M", fontsize=11)
    ax[0].legend(fontsize=8.5)
    for D_L, c in zip(DL_family, cols):
        r = [x for x in rows if x["study"] == "length" and x["D_L"] == D_L]
        ax[1].plot([x["M"] for x in r], [x["system_rate"] for x in r], "s-", color=c, lw=1.8, label=f"D_L = {D_L:.0f} m")
    ax[1].set_xlabel("segments M")
    ax[1].set_ylabel("system sum rate [bps/Hz]")
    ax[1].set_title("B. Length changes the best M (K = 1)", fontsize=11)
    ax[1].legend(fontsize=8.5)
    Ks = [r["K"] for r in cmp_rows]
    ax[2].plot(Ks, [r["epma"] for r in cmp_rows], "o-", color=MAROON, lw=2, label="EPMA")
    ax[2].plot(Ks, [r["orthogonal_B"] for r in cmp_rows], "s--", color=GREEN, lw=1.8, label="orthogonal, average power")
    ax[2].plot(Ks, [r["orthogonal_A"] for r in cmp_rows], "^:", color=GREY, lw=1.8, label="orthogonal, peak power")
    ax[2].set_xscale("log", base=2)
    ax[2].set_xticks(Ks)
    ax[2].set_xticklabels([str(k) for k in Ks], fontsize=8)
    ax[2].set_xlabel("users per segment K")
    ax[2].set_ylabel("system sum rate [bps/Hz]")
    ax[2].set_title(f"C. EPMA against orthogonal access, M = {M0}", fontsize=11)
    ax[2].legend(fontsize=8.5)
    fig.tight_layout()
    save_figure(fig, "segments.png")


if __name__ == "__main__":
    main()
