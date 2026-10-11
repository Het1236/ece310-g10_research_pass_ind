"""M2 initial evaluation: what does loading K users per segment buy, and how does it compare with orthogonal access?

Three things are computed.
  1. The crossover load K* = 1 / (2 rho L1) where the leakage equals the noise (K* table vs SNR).
  2. The gain of K users in a region over one user in a region, Gamma(K) = K (K* + 1) / (K* + K).
  3. The system sum rate of the proposed scheme against two orthogonal benchmarks:
       - system-wide time division: all M*K users take turns (EDMA Eq. 9), Version B (average power)
         and Version A (peak power).
       - the same segmented system with one user per segment per slot, M * R(1).
     The proposed scheme is also shown with the blockage windfall removed.

Run from the Code folder:
    python experiments/m2_loading_benchmark.py
"""
from __future__ import annotations

import copy
import csv

from m2_common import RESULTS_DIR, MAROON, GREY, BLUE, GREEN, ORANGE, save_figure, setup_matplotlib

from src import analytic
from src.config import load_params

RHO_DB = [80, 100, 110, 120]
K_TABLE = [1, 2, 5, 10, 16, 20, 32, 64]
K_MAX = 64


def with_snr(params: dict, rho_db: float) -> dict:
    q = copy.deepcopy(params)
    q["rho"] = 10 ** (rho_db / 10)
    q["phase3"]["N0"] = float(q["phase3"]["P"]) / q["rho"]
    return q


def main() -> None:
    base = load_params()
    M = float(base["M"])

    # 1. crossover load K*
    kstar_rows = []
    for db in RHO_DB:
        p = with_snr(base, db)
        L1 = analytic.L1_exact(p)
        kstar_rows.append({"rho_dB": db, "L1_exact": L1, "K_star": 1.0 / (2.0 * p["rho"] * L1)})
    print("K* (M = 3):")
    for r in kstar_rows:
        print(f"  {r['rho_dB']:>4} dB   K* = {r['K_star']:.3e}")
    with open(RESULTS_DIR / "kstar_vs_snr.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(kstar_rows[0].keys()))
        w.writeheader()
        w.writerows(kstar_rows)

    # 2 and 3. gain and benchmark comparison, K = 1 .. 64, three SNR values
    rows = []
    cross = {}
    for db in (100, 110, 120):
        p = with_snr(base, db)
        r1 = analytic.ergodic_rate_exact(1, p)[0]
        cross[db] = None
        for K in range(1, K_MAX + 1):
            rbar = analytic.ergodic_rate_exact(K, p)[0]
            row = {
                "rho_dB": db, "K": K,
                "epma_system": M * rbar,
                "epma_windfall_removed": M * analytic.R_tilde(K, p),
                "orth_system_B": analytic.tdma_system_rate("B", K, M, p, placement="u0"),
                "orth_system_A": analytic.tdma_system_rate("A", K, M, p, placement="u0"),
                "same_system_K1": M * r1,
                "gamma": analytic.gain_factor(K, p),
            }
            row["epma_over_orthB"] = row["epma_system"] / row["orth_system_B"]
            rows.append(row)
            if cross[db] is None and row["epma_system"] < row["orth_system_B"]:
                cross[db] = K
    with open(RESULTS_DIR / "loading_vs_orthogonal.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\nsystem sum rate at 120 dB, M = 3 [bps/Hz]")
    print(f"{'K':>3} {'proposed':>9} {'no windfall':>11} {'orth B':>7} {'orth A':>7} {'K=1 reuse':>9} {'ratio':>6} {'Gamma':>7}")
    for r in rows:
        if r["rho_dB"] == 120 and r["K"] in K_TABLE:
            print(f"{r['K']:>3} {r['epma_system']:>9.2f} {r['epma_windfall_removed']:>11.2f} {r['orth_system_B']:>7.2f} "
                  f"{r['orth_system_A']:>7.2f} {r['same_system_K1']:>9.2f} {r['epma_over_orthB']:>6.2f} {r['gamma']:>7.4f}")
    for db, k in cross.items():
        print(f"{db} dB: the proposed scheme falls below system-wide orthogonal (version B) from K = {k}")

    # figure
    plt = setup_matplotlib()
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.0))
    sel = [r for r in rows if r["rho_dB"] == 120]
    ks = [r["K"] for r in sel]
    ax[0].plot(ks, [r["epma_system"] for r in sel], "-", color=MAROON, lw=2, label="proposed (two layers)")
    ax[0].plot(ks, [r["epma_windfall_removed"] for r in sel], "--", color=GREY, lw=1.6, label="proposed, windfall removed")
    ax[0].plot(ks, [r["orth_system_B"] for r in sel], "-", color=BLUE, lw=1.8, label="system-wide TDMA, average power")
    ax[0].plot(ks, [r["orth_system_A"] for r in sel], ":", color=BLUE, lw=1.8, label="system-wide TDMA, peak power")
    ax[0].plot(ks, [r["same_system_K1"] for r in sel], "-.", color=GREEN, lw=1.6, label="same segments, one user per slot")
    ax[0].set_xlabel("users per segment K")
    ax[0].set_ylabel("system sum rate [bps/Hz]")
    ax[0].set_title("120 dB, M = 3", fontsize=11)
    ax[0].legend(fontsize=8.5, loc="upper right", bbox_to_anchor=(1.0, 0.93))

    for db, col in ((100, GREEN), (110, ORANGE), (120, MAROON)):
        s = [r for r in rows if r["rho_dB"] == db]
        ax[1].plot([r["K"] for r in s], [r["gamma"] for r in s], color=col, lw=2, label=f"{db} dB")
    ax[1].plot([1, K_MAX], [1, K_MAX], ":", color="black", lw=0.8, label="ideal gain K")
    ax[1].set_xlabel("users per segment K")
    ax[1].set_ylabel(r"gain $\Gamma(K)$ of loading")
    ax[1].set_yscale("log")
    ax[1].set_xscale("log")
    ax[1].legend(fontsize=9)
    fig.tight_layout()
    save_figure(fig, "loading_vs_orthogonal.png")


if __name__ == "__main__":
    main()
