"""M3 result 2: what is the antenna placement worth for the sum rate?

For K = 1, 2, 4, 8 and 16 users in a segment, 200 random drops each, I compare three ways to
place the pinching antenna (every user transmits at the peak power, so placement is the whole problem):
  - optimal: the certified best position (golden-section / K-start search, checked on a dense grid),
  - anchored: the antenna sits above the user with the strongest channel (the EDMA rule for K = 1),
  - centre: the antenna stays at the centre of the segment.
The interference floor moves with the antenna in every case.

It also checks three things that the report relies on:
  - the cost of the search (iterations against the tolerance),
  - that the search agrees with an exhaustive grid search,
  - that at equal total power uplink NOMA and boosted time sharing have the same sum rate.

Run from the Code folder:
    python experiments/m3_placement_rules.py            # 200 drops per K
    python experiments/m3_placement_rules.py --quick    # 30 drops per K
"""
from __future__ import annotations

import argparse
import csv

import numpy as np

from m3_common import RESULTS_DIR, MAROON, GREY, BLUE, GREEN, save_figure, setup_matplotlib

from src import floor_position, placement
from src.config import load_params

K_LIST = [1, 2, 4, 8, 16]


def mac_tie_deviation(rng: np.random.Generator, trials: int = 2000) -> float:
    """Largest gap between the NOMA sum rate and boosted time sharing at equal total power."""
    worst = 0.0
    for _ in range(trials):
        K = int(rng.integers(2, 9))
        g = 10.0 ** rng.uniform(-9, -6, K)
        Pk = rng.uniform(0.05, 1.0, K)
        F = 10.0 ** rng.uniform(-12, -9)
        noma = np.log2(1.0 + np.sum(Pk * g) / F)
        tau = Pk * g / np.sum(Pk * g)
        tdma = np.sum(tau * np.log2(1.0 + (Pk / tau) * g / F))
        worst = max(worst, abs(noma - tdma))
    return float(worst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    n_drops = 30 if args.quick else 200

    p = load_params()
    D_S, D_W, d, eta = float(p["D_S"]), float(p["D_W"]), float(p["d"]), float(p["eta"])
    P = float(p["phase4"]["P_max"])
    tol = float(p["phase4"]["gs_tol"])
    rng = np.random.default_rng(int(p["seed"]))
    boot = np.random.default_rng(7)
    xs_ref = np.linspace(0.0, D_S, 20001)

    rows, check_rows = [], []
    for K in K_LIST:
        Ffn = floor_position.floor_fn_absolute(K, p, 0.0)
        F_ref = np.array([Ffn(x) for x in xs_ref])
        opt, anc, cen, dev = [], [], [], []
        for i in range(n_drops):
            ux = rng.uniform(0.0, D_S, K)
            uy = rng.uniform(-D_W / 2.0, D_W / 2.0, K)
            obj = lambda x, ux=ux, uy=uy: placement.rate_at_position(x, ux, uy, eta, d, P, Ffn)
            _, f_opt, _ = placement.golden_section_kstart(obj, 0.0, D_S, ux, tol)
            x_anc = float(ux[int(np.argmin(uy ** 2 + d * d))])
            opt.append(f_opt)
            anc.append(obj(x_anc))
            cen.append(obj(D_S / 2.0))
            if i < 50:   # exhaustive check on the first 50 drops
                G = placement.collected_power(xs_ref, ux, uy, eta, d, P)
                f_ex = float(np.max(np.log2(1.0 + G / F_ref)))
                dev.append(abs(f_ex - f_opt) / f_ex)
        opt, anc, cen = map(np.array, (opt, anc, cen))
        idx = boot.integers(0, n_drops, (2000, n_drops))
        row = {"K": K, "drops": n_drops, "optimal": float(opt.mean()), "anchored": float(anc.mean()), "centre": float(cen.mean())}
        for name, X in (("anchored", anc), ("centre", cen)):
            row[f"{name}_deficit_pct"] = float(100.0 * (1.0 - X.mean() / opt.mean()))
            row[f"{name}_deficit_se"] = float(np.std(100.0 * (1.0 - X[idx].mean(axis=1) / opt[idx].mean(axis=1)), ddof=1))
        rows.append(row)
        check_rows.append({"K": K, "exhaustive_max_rel_dev": float(np.max(dev))})
        print(f"K={K:<2d}: optimal {row['optimal']:.3f}, anchored {row['anchored']:.3f} (-{row['anchored_deficit_pct']:.1f} % "
              f"+- {row['anchored_deficit_se']:.1f}), centre {row['centre']:.3f} (-{row['centre_deficit_pct']:.2f} % "
              f"+- {row['centre_deficit_se']:.2f}); search against exhaustive grid: {max(dev):.1e}", flush=True)

    with open(RESULTS_DIR / "placement_rules.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # cost of the search: iterations against the tolerance
    cost_rows = []
    rng2 = np.random.default_rng(11)
    for K in (1, 2, 4):
        Ffn = floor_position.floor_fn_absolute(K, p, 0.0)
        for eps in [float(e) for e in p["phase4"]["figure"]["complexity_eps_grid"]]:
            its = []
            for _ in range(40):
                ux = rng2.uniform(0.0, D_S, K)
                uy = rng2.uniform(-D_W / 2.0, D_W / 2.0, K)
                obj = lambda x, ux=ux, uy=uy: placement.rate_at_position(x, ux, uy, eta, d, P, Ffn)
                its.append(placement.golden_section_kstart(obj, 0.0, D_S, ux, eps)[2])
            cost_rows.append({"K": K, "tolerance_m": eps, "mean_iterations": float(np.mean(its))})
    with open(RESULTS_DIR / "search_cost.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["K", "tolerance_m", "mean_iterations"])
        w.writeheader()
        w.writerows(cost_rows)
    for K in (1, 2, 4):
        its = [r["mean_iterations"] for r in cost_rows if r["K"] == K]
        print(f"K={K}: mean iterations from tolerance 0.1 m to 1e-5 m: {its[0]:.1f} -> {its[-1]:.1f}  "
              f"(about {(its[-1] - its[0]) / 4:.1f} more per decade)")

    tie = mac_tie_deviation(np.random.default_rng(5))
    print(f"equal total power: largest gap between NOMA and boosted time sharing over 2000 random cases = {tie:.1e} bps/Hz")
    with open(RESULTS_DIR / "mac_tie.csv", "w", newline="", encoding="utf-8") as fh:
        fh.write("cases,largest_gap_bps_per_hz\n")
        fh.write(f"2000,{tie:.3e}\n")

    plt = setup_matplotlib()
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.0))
    Ks = [r["K"] for r in rows]
    ax[0].plot(Ks, [r["optimal"] for r in rows], "o-", color=MAROON, lw=2, label="certified optimal position")
    ax[0].plot(Ks, [r["centre"] for r in rows], "s--", color=BLUE, lw=1.6, label="fixed at the segment centre")
    ax[0].plot(Ks, [r["anchored"] for r in rows], "^:", color=GREEN, lw=1.6, label="above the strongest user")
    ax[0].set_xscale("log", base=2)
    ax[0].set_xticks(Ks)
    ax[0].set_xticklabels([str(k) for k in Ks])
    ax[0].set_xlabel("users per segment K")
    ax[0].set_ylabel("segment sum rate [bps/Hz]")
    ax[0].legend(fontsize=9)

    ax[1].errorbar(Ks, [r["anchored_deficit_pct"] for r in rows], yerr=[2 * r["anchored_deficit_se"] for r in rows], fmt="^:", color=GREEN, lw=1.6, label="above the strongest user")
    ax[1].errorbar(Ks, [r["centre_deficit_pct"] for r in rows], yerr=[2 * r["centre_deficit_se"] for r in rows], fmt="s--", color=BLUE, lw=1.6, label="fixed at the segment centre")
    ax[1].set_xscale("log", base=2)
    ax[1].set_yscale("log")
    ax[1].set_xticks(Ks)
    ax[1].set_xticklabels([str(k) for k in Ks])
    ax[1].set_xlabel("users per segment K")
    ax[1].set_ylabel("rate lost against the optimum [%]")
    ax[1].legend(fontsize=9)
    fig.tight_layout()
    save_figure(fig, "placement_rules.png")


if __name__ == "__main__":
    main()
