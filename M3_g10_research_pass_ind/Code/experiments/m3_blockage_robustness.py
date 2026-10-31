"""M3 result 4: what do blocked own links and real (random) interference cost?

The design model has two simplifications. Own links are always line of sight, and the interference
floor is its mean value. This script measures what each simplification costs.

  A. Own links blocked (home-LoS coin). Every user's own link is blocked with probability
     1 - exp(-phi r^2). The rate is averaged over the coin outside the log. I compare it with the
     always-LoS rate at the same antenna position.
  B. Real interference instead of the mean floor. The antenna position and powers of the certified
     max-min design are fixed. The interferers in the two neighbour segments are drawn at random and
     blocked at random. I report how often the minimum user rate falls under the design target.
  C. Blocked links that still leak a little (NLoS residual kappa). A blocked link carries kappa times
     its free-space gain instead of zero. I report the system rate and K* for kappa = off, -30, -20, -10 dB.

Run from the Code folder:
    python experiments/m3_blockage_robustness.py            # full run
    python experiments/m3_blockage_robustness.py --quick    # smaller samples
"""
from __future__ import annotations

import argparse
import csv

import numpy as np

from m3_common import RESULTS_DIR, MAROON, GREY, BLUE, GREEN, ORANGE, save_figure, setup_matplotlib

from src import analytic as A
from src import channel, floor_position, maxmin_global, placement, power_alloc
from src.config import load_params


def home_los_loss(p, K_list, n_drops, n_coins):
    D_S, D_W, d = float(p["D_S"]), float(p["D_W"]), float(p["d"])
    phi, eta = float(p["phi"]), float(p["eta"])
    P = float(p["phase4"]["P_max"])
    tol = float(p["phase4"]["gs_tol"])
    rows = []
    for K in K_list:
        F = floor_position.floor_fn_absolute(K, p, 0.0)
        loss, plos = [], []
        for i in range(n_drops):
            rng = np.random.default_rng(1000 + i)
            ux = rng.uniform(0.0, D_S, K)
            uy = rng.uniform(-0.5 * D_W, 0.5 * D_W, K)

            def obj(x):
                g = channel.path_gain((x - ux) ** 2 + uy ** 2 + d ** 2, eta)
                return float(power_alloc.segment_rate_from_powers(np.full(K, P), g, float(F(x))))

            x_star, r_full, _ = placement.golden_section_kstart(obj, 0.0, D_S, ux, tol)
            fl = float(F(x_star))
            r2 = (float(x_star) - ux) ** 2 + uy ** 2 + d ** 2
            g = channel.path_gain(r2, eta)
            pl = np.exp(-phi * r2)
            coins = (rng.random((n_coins, K)) < pl).astype(float)
            real = np.log2(1.0 + (coins * P * g).sum(axis=1) / fl).mean()
            loss.append(100.0 * (r_full - real) / r_full)
            plos.append(pl.mean())
        loss = np.array(loss)
        rows.append({"K": K, "mean_p_los": float(np.mean(plos)), "loss_pct": float(loss.mean()),
                     "loss_se": float(loss.std(ddof=1) / np.sqrt(n_drops))})
        print(f"A  K={K}: mean P(LoS) {rows[-1]['mean_p_los']:.3f}, own-link blockage costs "
              f"{rows[-1]['loss_pct']:.2f} % +- {rows[-1]['loss_se']:.2f}", flush=True)
    return rows


def realised_interference(p, n):
    cfg = p["phase4"]["figure"]
    D_S, D_W, d, eta = float(p["D_S"]), float(p["D_W"]), float(p["d"]), float(p["eta"])
    phi, P, N0 = float(p["phi"]), float(p["phase4"]["P_max"]), float(p["phase3"]["N0"])
    sep = float(cfg["x_sep_frac"]) * D_S
    c = D_S / 2.0
    B_F = floor_position.floor_logderiv_bound(p)
    rng = np.random.default_rng(2026)
    rows = []
    for K in (2, 4):
        ux = np.linspace(-0.5, 0.5, K) * sep + c
        uy = np.zeros(K)
        F = floor_position.floor_fn_absolute(K, p, 0.0)
        g = maxmin_global.maxmin_global_placement(ux, uy, 0.0, D_S, d, eta, F, "A", P_max=P, floor_logderiv=B_F)
        x = float(g["x"])
        gains = channel.path_gain((x - ux) ** 2 + uy ** 2 + d ** 2, eta)
        t_star, Pstar, _ = power_alloc.maxmin_bisection(gains, float(F(x)), "A", P_max=P)
        left = rng.uniform(-D_S, 0.0, (n, K))
        right = rng.uniform(D_S, 2 * D_S, (n, K))
        ix = np.concatenate([left, right], axis=1)
        iy = rng.uniform(-D_W / 2, D_W / 2, ix.shape)
        r2 = (x - ix) ** 2 + iy ** 2 + d ** 2
        coin = rng.random(ix.shape) < np.exp(-phi * r2)
        fl = N0 + (coin * P * channel.path_gain(r2, eta)).sum(axis=1)
        order = power_alloc.sic_order(gains)
        sp = Pstar * gains
        minr = np.full(n, np.inf)
        for i, k in enumerate(order):
            later = order[i + 1:]
            minr = np.minimum(minr, np.log2(1.0 + sp[k] / (sp[later].sum() + fl)))
        short = np.maximum(t_star - minr, 0.0)
        p_out = float((minr < t_star - 1e-9).mean())
        rows.append({"K": K, "target": float(t_star), "x_star": x, "p_below": p_out,
                     "mean_shortfall": float(short.mean()),
                     "shortfall_given_below": float(short[short > 0].mean()) if (short > 0).any() else 0.0,
                     "median_min_rate": float(np.median(minr))})
        r = rows[-1]
        print(f"B  K={K}: design target {r['target']:.3f} at x={x:.2f} m; real min rate falls under it in "
              f"{100 * r['p_below']:.1f} % of draws; mean shortfall {r['mean_shortfall']:.3f} bit "
              f"({r['shortfall_given_below']:.3f} when it falls short)", flush=True)
    return rows


def nlos_residual(p):
    eta, phi, d, D_S, D_W, delta, P, N0 = A._phys(p)
    rho = P / N0
    M = float(p["M"])
    r2, w = A._position_grid(D_S - delta, 2.0 * D_S - delta, D_W, d)
    orig = A.h_transform
    rows = []
    for kdb in (None, -30.0, -20.0, -10.0):
        kap = 0.0 if kdb is None else 10 ** (kdb / 10)

        def h_k(z, params, kap=kap):
            pl = np.exp(-phi * r2)
            return float(np.sum(w * (pl * np.exp(-z * P * eta / r2) + (1 - pl) * np.exp(-z * kap * P * eta / r2))))

        A.h_transform = h_k
        try:
            R1 = A.ergodic_rate_exact(1, p)[0]
            R5 = A.ergodic_rate_exact(5, p)[0]
        finally:
            A.h_transform = orig
        L1k = float(np.sum(w * eta / r2 * (np.exp(-phi * r2) + kap * (1 - np.exp(-phi * r2)))))
        pb = float(np.sum(w * (1 - np.exp(-phi * r2))))
        rows.append({"kappa_dB": "off" if kdb is None else f"{kdb:.0f}", "system_rate_K1": M * R1, "system_rate_K5": M * R5,
                     "K_star": 1.0 / (2 * rho * L1k), "p_all_blocked_K1": pb ** 2, "p_all_blocked_K5": pb ** 10})
        r = rows[-1]
        print(f"C  kappa={r['kappa_dB']:>4}: system rate K=1 {r['system_rate_K1']:.2f}, K=5 {r['system_rate_K5']:.2f}; "
              f"K* {r['K_star']:.2e}; P(all interferers blocked) K=1 {r['p_all_blocked_K1']:.3f}, K=5 {r['p_all_blocked_K5']:.3f}", flush=True)
    tdma = A.tdma_system_rate("B", 1, int(p["M"]), p)
    print(f"C  reference: orthogonal access (average power) system rate {tdma:.2f}")
    return rows, tdma


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
    n_drops, n_coins, n_real = (40, 60, 20000) if args.quick else (200, 200, 200000)

    a = home_los_loss(p, [1, 2, 4, 8], n_drops, n_coins)
    b = realised_interference(p, n_real)
    c, tdma = nlos_residual(p)
    write_csv("own_link_blockage.csv", a)
    write_csv("realised_interference.csv", b)
    write_csv("nlos_residual.csv", c)

    plt = setup_matplotlib()
    fig, ax = plt.subplots(1, 3, figsize=(11.6, 3.8))
    ax[0].errorbar([r["K"] for r in a], [r["loss_pct"] for r in a], yerr=[2 * r["loss_se"] for r in a],
                   fmt="o-", color=MAROON, lw=2, capsize=3)
    ax[0].set_xlabel("users per segment K")
    ax[0].set_ylabel("rate lost to blocked own links [%]")
    ax[0].set_title("A. Own links blocked", fontsize=11)

    xs = np.arange(len(b))
    ax[1].bar(xs, [100 * r["p_below"] for r in b], 0.5, color=BLUE)
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels([f"K = {r['K']}" for r in b])
    ax[1].set_ylabel("draws below the design target [%]")
    ax[1].set_title("B. Real interference", fontsize=11)

    lab = [r["kappa_dB"] for r in c]
    xk = np.arange(len(c))
    ax[2].plot(xk, [r["system_rate_K1"] for r in c], "o-", color=MAROON, lw=2, label="EPMA, K = 1")
    ax[2].plot(xk, [r["system_rate_K5"] for r in c], "s--", color=BLUE, lw=1.8, label="EPMA, K = 5")
    ax[2].axhline(tdma, color=GREEN, ls=":", lw=2, label="orthogonal access, average power")
    ax[2].set_xticks(xk)
    ax[2].set_xticklabels(lab)
    ax[2].set_xlabel("leak through a blocked link [dB]")
    ax[2].set_ylabel("system rate [bps/Hz]")
    ax[2].set_title("C. Blocked links that leak", fontsize=11)
    ax[2].legend(fontsize=8.5)
    fig.tight_layout()
    save_figure(fig, "blockage_robustness.png")


if __name__ == "__main__":
    main()
