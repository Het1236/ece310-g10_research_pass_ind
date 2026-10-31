"""M3 result 3: max-min fairness inside a segment.

Layout (the same for every K): K users spread evenly over one third of the segment width around the
centre, all on the waveguide axis except the last user, who is moved off axis by y2 metres. A larger y2
means a weaker channel for that user. The service region ends at y2 = D_W / 2 = 5 m.

For each y2 I compute the best possible minimum user rate (the max-min floor) with
  - the certified global design of the final method (best position and best powers),
  - the antenna fixed at the segment centre with the best powers,
  - system-wide style orthogonal access: users take turns, with the antenna at its best position for the
    chosen time sharing (peak-power version),
  - a naive alternating optimisation (powers, then position, repeated) started from five different
    positions. I report its best and its worst start.
The interference floor moves with the antenna in every case.

Run from the Code folder:
    python experiments/m3_fairness.py            # K = 2 and 4, 15 values of y2
    python experiments/m3_fairness.py --quick    # fewer values of y2
"""
from __future__ import annotations

import argparse
import csv

import numpy as np

from m3_common import RESULTS_DIR, MAROON, GREY, BLUE, GREEN, ORANGE, save_figure, setup_matplotlib

from src import altopt, analytic, channel, floor_position, maxmin_global, power_alloc
from src.config import load_params

K_FAMILY = [2, 4]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    p = load_params()
    cfg = p["phase4"]["figure"]
    D_S, d, eta = float(p["D_S"]), float(p["d"]), float(p["eta"])
    P_max = float(p["phase4"]["P_max"])
    sep = float(cfg["x_sep_frac"]) * D_S
    c = D_S / 2.0
    B_F = floor_position.floor_logderiv_bound(p)
    y2_grid = np.array([float(v) for v in cfg["y2_grid"]])
    if args.quick:
        y2_grid = y2_grid[::3]
    n_tdma = 401 if args.quick else 2001
    print(f"B_F = {B_F:.4f} per metre; {len(y2_grid)} values of y2; time-sharing grid {n_tdma} positions")

    def gains(x, ux, uy):
        return channel.path_gain((np.asarray(x) - ux) ** 2 + uy ** 2 + d ** 2, eta)

    def layout(K, y2):
        ux = (np.linspace(-0.5, 0.5, K) * sep + c) if K > 1 else np.array([c])
        uy = np.zeros(K)
        uy[-1] = float(y2)
        return ux, uy

    rows = []
    anchors = (c - D_S, c + D_S)
    for K in K_FAMILY:
        F_x = floor_position.floor_fn_absolute(K, p, 0.0)
        fl_frozen = float(p["phase3"]["N0"]) + 2.0 * K * float(p["phase3"]["P"]) * analytic.L1_exact(p)
        for y2 in y2_grid:
            ux, uy = layout(K, y2)
            g = maxmin_global.maxmin_global_placement(ux, uy, 0.0, D_S, d, eta, F_x, "A", P_max=P_max, floor_logderiv=B_F)
            centre = power_alloc.maxmin_bisection(gains(c, ux, uy), F_x(c), "A", P_max=P_max)[0]
            tdma = max(power_alloc.tdma_maxmin(gains(x, ux, uy), F_x(x), "A", P_max=P_max)
                       for x in np.linspace(0.0, D_S, n_tdma))
            ms = altopt.ao_multistart(ux, uy, 0.0, D_S, anchors, "maxmin", "A", p, n_random=2, floor=fl_frozen)
            scored = [power_alloc.maxmin_bisection(gains(float(r["x_star"]), ux, uy), F_x(float(r["x_star"])), "A", P_max=P_max)[0]
                      for r in ms["results"].values()]
            rows.append({"K": K, "y2": float(y2), "certified": g["f"], "x_star": g["x"], "gap": g["gap"],
                         "evaluations": g["n_eval"], "centre": centre, "orthogonal": tdma,
                         "naive_best": max(scored), "naive_worst": min(scored)})
            print(f"K={K} y2={y2:4.1f}: certified {g['f']:.3f} at x={g['x']:.2f} m, centre {centre:.3f}, orthogonal {tdma:.3f}, "
                  f"naive best {max(scored):.3f} worst {min(scored):.3f}", flush=True)

    with open(RESULTS_DIR / "fairness.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    half = float(p["D_W"]) / 2.0
    for K in K_FAMILY:
        r = [x for x in rows if x["K"] == K]
        r0 = r[0]
        edge = min(r, key=lambda x: abs(x["y2"] - half))
        print(f"K={K}: certified floor {r0['certified']:.3f} at y2 = 0 and {edge['certified']:.3f} at the service edge; "
              f"orthogonal {r0['orthogonal']:.3f} -> {edge['orthogonal']:.3f}; gap at the edge {edge['certified'] - edge['orthogonal']:.3f} bps/Hz; "
              f"fixed centre costs up to {max(x['certified'] - x['centre'] for x in r):.3f} bps/Hz; "
              f"naive worst start loses up to {max(x['certified'] - x['naive_worst'] for x in r):.3f} bps/Hz; "
              f"largest certified gap {max(x['gap'] for x in r):.1e}; mean evaluations {np.mean([x['evaluations'] for x in r]):.0f}")

    plt = setup_matplotlib()
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.0), sharey=False)
    for a, K in zip(ax, K_FAMILY):
        r = [x for x in rows if x["K"] == K]
        y = [x["y2"] for x in r]
        a.plot(y, [x["certified"] for x in r], "o-", color=MAROON, lw=2, label="certified design")
        a.plot(y, [x["centre"] for x in r], "s--", color=BLUE, lw=1.6, label="antenna at the centre")
        a.plot(y, [x["orthogonal"] for x in r], "^:", color=GREEN, lw=1.8, label="orthogonal access")
        a.plot(y, [x["naive_best"] for x in r], "x", color=ORANGE, ms=8, label="naive alternation, best start")
        a.plot(y, [x["naive_worst"] for x in r], "+", color=GREY, ms=9, label="naive alternation, worst start")
        a.axvline(half, color="black", lw=0.8, ls=":")
        a.set_xlabel("off-axis offset of the last user y2 [m]")
        a.set_ylabel("max-min floor [bps/Hz]")
        a.set_title(f"K = {K} users per segment", fontsize=11)
        a.legend(fontsize=8.2, loc="lower left")
    fig.tight_layout()
    save_figure(fig, "fairness.png")


if __name__ == "__main__":
    main()
