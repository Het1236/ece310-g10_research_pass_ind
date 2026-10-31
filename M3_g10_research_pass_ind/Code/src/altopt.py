"""Alternating optimization (AO): placement <-> power, per segment (Phase 4, P4-4).

Sources: GT4 = Phase-4 ground truth, CLAUDE.md Sec. 9 (AO item; each function cites it).
Pure ORCHESTRATION of the certified sub-solvers -- placement.golden_section_kstart
(P4-1.5) and power_alloc.power_lp / maxmin_bisection (P4-2/3). NO new physics; gains
via channel.path_gain, rates via power_alloc helpers.

Block coordinate ascent (GT4 AO):
  P-step: fix position, solve the power block (LP for "sumrate" [+ optional QoS],
          maxmin_bisection for "maxmin") -- the convex block, solved exactly (LP) or
          to bisection tolerance;
  x-step: fix powers, maximize the joint objective in x, ROUTED BY OBJECTIVE
          (P4-4.5): "sumrate" -> placement.golden_section_kstart (the weighted
          collected power is still a sum of K Lorentzian bumps, so the basin
          decomposition is exact - UNCHANGED); "maxmin" -> placement.scan_refine_max
          (the min-rate lower envelope has crossing/balance maxima that are NOT
          user-anchored - the ⚠️ P4-4 finding - so K-start is invalid there; the
          coarse-scan+refine solver is exhaustive-certified in Gate A-maxmin).
MONOTONICITY: each accepted step maximizes its block, so the joint objective is
non-decreasing -- enforced by SAFEGUARDED ACCEPTANCE (a block result is adopted ONLY
if it does not degrade the incumbent; finite sub-solver tolerances gs_tol/t_tol can
return an epsilon-worse point once converged) plus a HARD runtime assert on the
accepted history (any decrease is a bug). A raw solver result that dips more than
_BLOCK_DIP_TOL below the incumbent is a WIRING BUG and raises immediately -- the
safeguard never hides real regressions.
CONVERGENCE: monotone non-decreasing + bounded above (finite power, gains capped by
eta/d^2 => finite rate) => converges. The limit is a COORDINATE-WISE / KKT STATIONARY
point of the NON-CONVEX joint problem (position enters non-concavely, x*P bilinear) --
NOT a certified best-possible point. Multi-start agreement (``ao_multistart``) is
EMPIRICAL evidence, never proof.
SUM-RATE DECOUPLING (GT4): under "sumrate" (Version A, no QoS) the power block's
optimum (all users at the cap) is position-independent, so AO converges in one
improving pass (n_iter <= 2 including the confirming pass). Coupling bites under
"maxmin", where the power shaping depends on the gains.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from src import analytic, channel, placement, power_alloc

# Raw-solver dip guard (wiring-bug detector): dominated by maxmin_bisection's
# t_tol = 1e-4 floor-certification slack; gs_tol dips are O(1e-7). A block solver
# returning more than this below the incumbent objective is a real bug, not tolerance.
_BLOCK_DIP_TOL: float = 5e-4

_STRING_INITS = ("midpoint", "over_strongest", "over_weakest", "random")


def _gains_at(
    x: float, users_x: np.ndarray, users_y: np.ndarray, eta: float, d: float
) -> np.ndarray:
    """g_{m,k}(x) = eta/((x-x_k)^2 + y_k^2 + d^2) via channel.path_gain (GT4 channel gain)."""
    return channel.path_gain((x - users_x) ** 2 + users_y ** 2 + d * d, eta)


def _joint_objective(
    x: float,
    P: np.ndarray,
    users_x: np.ndarray,
    users_y: np.ndarray,
    eta: float,
    d: float,
    floor: float,
    objective: str,
) -> float:
    """Joint AO objective f(x, P). Source: GT4 AO item.

    "sumrate": R_m^sum = log2(1 + sum_k P_k g_k(x)/floor) (GT4 true rate);
    "maxmin":  min_k R_{m,k} with the strongest-first SIC order recomputed at the
    actual gains g(x) (the production order convention; P4-2 locked).
    """
    g = _gains_at(x, users_x, users_y, eta, d)
    if objective == "sumrate":
        return power_alloc.segment_rate_from_powers(P, g, floor)
    return float(np.min(power_alloc.per_user_rates(P, g, floor)))


def _resolve_init(
    init: str | float,
    users_x: np.ndarray,
    users_y: np.ndarray,
    d: float,
    seg_lo: float,
    seg_hi: float,
    anchors: tuple[float, float],
    rng: np.random.Generator,
) -> float:
    """Initial PA position. "midpoint" = EDMA Remark 8 anchor midpoint (clamped);
    "over_strongest"/"over_weakest" = over the user with the largest/smallest peak
    gain eta/C_k (C_k = y_k^2 + d^2); "random" = uniform over the segment; a float is
    an explicit start (used by ao_multistart's random inits)."""
    if isinstance(init, (int, float)) and not isinstance(init, bool):
        return float(min(max(float(init), seg_lo), seg_hi))
    if init == "midpoint":
        return placement.midpoint_placement(anchors[0], anchors[1], seg_lo, seg_hi)
    C = users_y ** 2 + d * d
    if init == "over_strongest":
        return float(users_x[int(np.argmin(C))])
    if init == "over_weakest":
        return float(users_x[int(np.argmax(C))])
    if init == "random":
        return float(rng.uniform(seg_lo, seg_hi))
    raise ValueError(f"init must be a float or one of {_STRING_INITS}, got {init!r}")


def ao_segment(
    users_x: np.ndarray,
    users_y: np.ndarray,
    seg_lo: float,
    seg_hi: float,
    anchors: tuple[float, float],
    objective: str,
    power_model: str,
    params: Mapping[str, Any],
    init: str | float = "midpoint",
    max_iter: int = 50,
    tol: float = 1e-6,
    R_min: float | None = None,
    floor: float | None = None,
) -> dict[str, Any]:
    """Per-segment AO: alternate P-step and x-step until the per-iteration improvement
    is <= tol. Source: GT4 AO item (block coordinate ascent).

    The accepted objective sequence is NON-DECREASING by safeguarded acceptance (see
    module docstring) and hard-asserted; monotone + bounded => converges to a
    COORDINATE-WISE / KKT STATIONARY point of the non-convex joint problem -- no
    stronger optimality is claimed anywhere. R_min applies to "sumrate" only (QoS);
    "maxmin" needs no R_min (it maximizes the floor, GT4 max-min item). `floor`
    defaults to the Phase-3 mean interference constant N0 + 2K*P*L1_exact (the
    Phase-2-licensed near-constant floor). Version-B budget P_tot = K*P_max
    (confirmed). Returns {x_star, P_star, obj, n_iter, obj_history, init, status}.
    """
    if objective not in ("sumrate", "maxmin"):
        raise ValueError(f"objective must be 'sumrate' or 'maxmin', got {objective!r}")
    if objective == "maxmin" and R_min is not None:
        raise ValueError("maxmin needs NO R_min (it MAXIMIZES the floor; GT4)")
    users_x = np.asarray(users_x, dtype=float)
    users_y = np.asarray(users_y, dtype=float)
    K = users_x.size
    eta, d = float(params["eta"]), float(params["d"])
    gs_tol = float(params["phase4"]["gs_tol"])
    P_max = float(params["phase4"]["P_max"])
    P_tot = K * P_max                                   # Version-B budget (confirmed)
    if floor is None:
        floor = float(params["phase3"]["N0"]) + 2.0 * K * float(
            params["phase3"]["P"]) * analytic.L1_exact(params)
    rng = np.random.default_rng(int(params["seed"]))
    lp_kw = {"P_max": P_max} if power_model == "A" else {"P_tot": P_tot}

    x = _resolve_init(init, users_x, users_y, d, seg_lo, seg_hi, anchors, rng)
    P = np.full(K, P_max if power_model == "A" else P_tot / K)   # uniform start
    f_cur = _joint_objective(x, P, users_x, users_y, eta, d, floor, objective)
    history = [f_cur]

    def _accept(f_new: float, label: str, n: int) -> bool:
        """Safeguarded acceptance + wiring-bug guard + hard monotonicity assert."""
        nonlocal f_cur
        if f_new < f_cur - _BLOCK_DIP_TOL:
            raise AssertionError(
                f"AO {label}-step at iter {n} degraded the objective beyond solver "
                f"tolerance: {f_new} < incumbent {f_cur} - {_BLOCK_DIP_TOL} -- wiring bug")
        accepted = f_new >= f_cur
        if accepted:
            f_cur = f_new
        history.append(f_cur)
        assert history[-1] >= history[-2] - 1e-9, (
            f"AO accepted history decreased at iter {n} ({label}-step): "
            f"{history[-2]} -> {history[-1]} -- bug")
        return accepted

    status = "max_iter"
    n_done = 0
    for n in range(1, max_iter + 1):
        f_start = f_cur
        # ---- P-step (convex block: LP exact / bisection to t_tol) ----
        g = _gains_at(x, users_x, users_y, eta, d)
        if objective == "sumrate":
            P_new, _, lp_status = power_alloc.power_lp(
                g, floor, power_model, R_min=R_min, **lp_kw)
            if lp_status != "optimal":
                raise ValueError(
                    f"P-step LP {lp_status} at x={x:.4f} (R_min={R_min}); choose a "
                    f"feasible R_min -- AO does not silently drop QoS floors")
        else:
            _, P_new, _ = power_alloc.maxmin_bisection(g, floor, power_model, **lp_kw)
        f_new = _joint_objective(x, P_new, users_x, users_y, eta, d, floor, objective)
        if _accept(f_new, "P", n):
            P = P_new
        # ---- x-step (routed by objective, P4-4.5: kstart exact for the sum-of-bumps
        # sumrate; scan+refine for the non-user-anchored min-rate envelope) ----
        def fx(xx: float) -> float:
            return _joint_objective(xx, P, users_x, users_y, eta, d, floor, objective)

        if objective == "sumrate":
            x_new, f_xnew, _ = placement.golden_section_kstart(
                fx, seg_lo, seg_hi, users_x, gs_tol)
        else:
            n_scan = int(params["phase4"]["maxmin_scan_points"])
            x_new, f_xnew, _ = placement.scan_refine_max(fx, seg_lo, seg_hi, n_scan, gs_tol)
        if _accept(f_xnew, "x", n):
            x = x_new
        n_done = n
        if f_cur - f_start <= tol:
            status = "converged"
            break

    return {"x_star": x, "P_star": P, "obj": f_cur, "n_iter": n_done,
            "obj_history": history, "init": init, "status": status}


def ao_multistart(
    users_x: np.ndarray,
    users_y: np.ndarray,
    seg_lo: float,
    seg_hi: float,
    anchors: tuple[float, float],
    objective: str,
    power_model: str,
    params: Mapping[str, Any],
    n_random: int = 2,
    **ao_kw: Any,
) -> dict[str, Any]:
    """Multi-start AO: run ao_segment from the named inits + n_random uniform starts;
    return the best plus ALL final objectives and their spread. Source: GT4 AO item.

    The spread (max - min of the final objectives) is the EMPIRICAL local-optimum
    diagnostic: near-zero spread is evidence -- never proof -- that the reached KKT
    stationary point is not init-dependent. A non-zero spread is a REAL finding
    (report it, do not hide it); the joint problem is non-convex and AO carries no
    stronger guarantee (GT4: no code may claim more than KKT stationarity).
    """
    rng = np.random.default_rng(int(params["seed"]))
    starts: list[tuple[str, str | float]] = [
        ("midpoint", "midpoint"),
        ("over_strongest", "over_strongest"),
        ("over_weakest", "over_weakest"),
    ] + [(f"random_{i}", float(rng.uniform(seg_lo, seg_hi))) for i in range(n_random)]

    results = {label: ao_segment(users_x, users_y, seg_lo, seg_hi, anchors, objective,
                                 power_model, params, init=val, **ao_kw)
               for label, val in starts}
    finals = {label: r["obj"] for label, r in results.items()}
    best_label = max(finals, key=finals.get)
    return {"best": results[best_label], "best_init": best_label,
            "all_final_objs": finals,
            "spread": max(finals.values()) - min(finals.values()),
            "results": results}


def ao_all_segments(
    users_x: np.ndarray,
    users_y: np.ndarray,
    objective: str,
    power_model: str,
    params: Mapping[str, Any],
    multistart: bool = False,
    **ao_kw: Any,
) -> list[dict[str, Any]]:
    """Independent AO per segment on the absolute grid m = 0..M-1. Source: GT4
    separability (uplink gift; segments are NOT coupled -- a segment's rate depends
    only on its own antenna and users, interference is the constant floor).

    users_x/users_y shape (M, K), absolute coordinates. Anchors for the midpoint init
    are the neighbouring segment centres (centre +- D_S; virtual at the edges).
    Returns the per-segment result list (ao_multistart results when multistart=True).
    """
    users_x = np.asarray(users_x, dtype=float)
    users_y = np.asarray(users_y, dtype=float)
    M = users_x.shape[0]
    D_S = float(params["D_L"]) / M
    out = []
    for m in range(M):
        lo, hi = m * D_S, (m + 1) * D_S
        centre = 0.5 * (lo + hi)
        anchors = (centre - D_S, centre + D_S)
        if multistart:
            out.append(ao_multistart(users_x[m], users_y[m], lo, hi, anchors,
                                     objective, power_model, params, **ao_kw))
        else:
            out.append(ao_segment(users_x[m], users_y[m], lo, hi, anchors,
                                  objective, power_model, params, **ao_kw))
    return out
