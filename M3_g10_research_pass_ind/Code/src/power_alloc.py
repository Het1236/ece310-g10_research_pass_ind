"""Per-segment uplink power allocation LP (Phase 4, P4-2).

Sources: GT4 = Phase-4 ground truth, CLAUDE.md Sec. 9 (each function cites its item).

Scope (P4-2 locked): PA positions are FIXED (from P4-1.5 placement or config), so the
channel gains g_{m,k} are CONSTANTS here; the per-user powers P_{m,k} are the variables.
Per-segment only (GT4 separability) -- segments are never coupled. The SIC decode order
is FIXED strongest-gain-first (``sic_order``) and passed explicitly to the constraint
builder; the QoS linearization is valid ONLY for a fixed order (GT4 power allocation).
Order and power are NEVER jointly optimized.

REVIEWER-HONESTY INVARIANT (GT4 ⚠, enforced by Gate E in tests/test_power_alloc.py):
at EQUAL TOTAL POWER the uplink NOMA sum rate equals the (boosted, optimally time-shared)
TDMA sum rate -- the sum rate sees only the total collected power sum_k P_k g_k. Nothing
in this module may be presented as a sum-rate win over TDMA at equal total power; the
NOMA wins are users-served / per-user / fairness / latency.

Power models (Phase-3 switch REUSED, no new convention):
  * Version "A" (per-user PEAK): 0 <= P_k <= P_max each. phase3's fixed P = 1 W per user
    IS this peak (config phase4.P_max = 1.0). Headline model.
  * Version "B" (shared TOTAL budget): sum_k P_k <= P_tot (P_tot = K*P_max derived by
    callers unless the user overrides -- flagged in params.yaml).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linprog, minimize


def qos_to_gamma(R_min: float) -> float:
    """QoS rate floor -> SINR floor: gamma = 2^{R_min} - 1. Source: GT4 power allocation
    (R_{m,k} >= R_min <=> SINR_{m,k} >= gamma, log2(1+s) monotone)."""
    if R_min < 0.0:
        raise ValueError(f"R_min must be >= 0, got {R_min}")
    return 2.0 ** R_min - 1.0


def sic_order(gains: np.ndarray) -> np.ndarray:
    """FIXED SIC decode order: users sorted by DECREASING gain g_{m,k}. Source: GT4
    power allocation (P4-2 locked decision).

    Returns index array `order` such that gains[order] is non-increasing;
    order[i] = original index of the user decoded i-th. Stable sort so ties keep the
    original relative order (deterministic). The QoS wall below is valid ONLY for this
    fixed order.
    """
    gains = np.asarray(gains, dtype=float)
    return np.argsort(-gains, kind="stable")


def build_qos_constraints(
    gains: np.ndarray,
    order: np.ndarray,
    gamma: float,
    floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """QoS floors as linear walls A @ P >= b in the FIXED decode order. Source: GT4
    power allocation (QoS linearization).

    Derivation (uplink SIC, user decoded i-th sees only the not-yet-decoded j > i as
    interference; floor = I_m + N0 constant, Phase-2 licence):
        R_k >= R_min  <=>  log2(1 + SINR_k) >= R_min
                      <=>  SINR_k = P_k g_k / (sum_{l>k} P_l g_l + floor) >= gamma
                      <=>  P_k g_k - gamma * sum_{l>k} P_l g_l >= gamma * floor.
    VALID ONLY FOR THE FIXED ORDER `order` (strongest-gain-first, ``sic_order``); a
    different order changes which l count as "later" and hence every wall.

    Rows/columns: row i is the wall of the user decoded i-th; columns index the
    ORIGINAL user numbering (A @ P uses P in original order). b[i] = gamma*floor.
    """
    gains = np.asarray(gains, dtype=float)
    order = np.asarray(order, dtype=int)
    K = gains.size
    A = np.zeros((K, K))
    for i in range(K):
        A[i, order[i]] = gains[order[i]]
        A[i, order[i + 1:]] = -gamma * gains[order[i + 1:]]
    b = np.full(K, gamma * floor)
    return A, b


def power_lp(
    gains: np.ndarray,
    floor: float,
    power_model: str,
    P_max: float | None = None,
    P_tot: float | None = None,
    R_min: float | None = None,
) -> tuple[np.ndarray | None, float, str]:
    """Per-segment power LP: maximize the collected power sum_k P_k g_k. Source: GT4
    power allocation (sum-rate objective LINEAR in power => the power step is an LP;
    argmax-equivalent to the true rate log2(1 + sum P g / floor), log monotone).

    Constraints: P >= 0; QoS walls (``build_qos_constraints``, only if R_min is given)
    in the FIXED strongest-gain-first order (``sic_order``, computed once here and
    passed explicitly); and per `power_model`:
      "A": 0 <= P_k <= P_max (per-user peak, REQUIRED);
      "B": sum_k P_k <= P_tot (shared budget, REQUIRED).
    Solver: scipy.optimize.linprog (HiGHS); rows pre-scaled by max(g) so coefficients
    are O(1) at physical gains (~1e-8). NOTE (GT4 ⚠): the optimal SUM rate at equal
    total power TIES TDMA -- this LP is never a sum-rate-win claim.

    Returns (P_star, obj, status): P_star in ORIGINAL user order and obj = collected
    power sum P_star*g at the optimum with status "optimal"; (None, nan, "infeasible")
    when the QoS floors cannot be met under the power constraints (returned CLEANLY,
    no exception); (None, nan, "solver:<code>") for any other solver outcome.
    """
    if power_model not in ("A", "B"):
        raise ValueError(f"power_model must be 'A' or 'B', got {power_model!r}")
    gains = np.asarray(gains, dtype=float)
    if gains.ndim != 1 or gains.size == 0 or np.any(gains <= 0.0):
        raise ValueError("gains must be a non-empty 1-D array of positive values")
    if floor <= 0.0:
        raise ValueError(f"floor = I_m + N0 must be > 0, got {floor}")
    K = gains.size
    g_scale = float(gains.max())            # row scaling: O(1) coefficients for HiGHS

    A_ub_rows, b_ub_vals = [], []
    if R_min is not None and R_min > 0.0:
        order = sic_order(gains)
        A_qos, b_qos = build_qos_constraints(gains, order, qos_to_gamma(R_min), floor)
        A_ub_rows.append(-A_qos / g_scale)  # A@P >= b  ->  -A@P <= -b
        b_ub_vals.append(-b_qos / g_scale)

    if power_model == "A":
        if P_max is None:
            raise ValueError("Version A requires P_max (per-user peak)")
        bounds = [(0.0, float(P_max))] * K
    else:
        if P_tot is None:
            raise ValueError("Version B requires P_tot (shared budget)")
        bounds = [(0.0, None)] * K
        A_ub_rows.append(np.ones((1, K)))
        b_ub_vals.append(np.array([float(P_tot)]))

    A_ub = np.vstack(A_ub_rows) if A_ub_rows else None
    b_ub = np.concatenate(b_ub_vals) if b_ub_vals else None

    res = linprog(c=-gains / g_scale, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if res.status == 2:
        return None, math.nan, "infeasible"
    if res.status != 0:
        return None, math.nan, f"solver:{res.status}"
    P_star = np.asarray(res.x, dtype=float)
    return P_star, float(gains @ P_star), "optimal"


def per_user_rates(
    P: np.ndarray,
    gains: np.ndarray,
    floor: float,
    order: np.ndarray | None = None,
) -> np.ndarray:
    """Per-user uplink SIC rates R_k = log2(1 + P_k g_k/(sum_{l>k} P_l g_l + floor)).

    Source: GT4 power allocation (the exact quantity the QoS wall linearizes) -- kept
    NEXT to ``build_qos_constraints`` so the SINR definition lives in one place; used
    by Gate B to recompute rates from powers rather than trusting the LP. `order`
    defaults to the fixed ``sic_order``. Returned in ORIGINAL user indexing. The rates
    telescope: sum_k R_k = log2(1 + sum_k P_k g_k / floor) (Phase-1 identity).
    """
    P = np.asarray(P, dtype=float)
    gains = np.asarray(gains, dtype=float)
    order = sic_order(gains) if order is None else np.asarray(order, dtype=int)
    rates = np.empty(gains.size)
    for i, k in enumerate(order):
        later = order[i + 1:]
        interf = float(np.sum(P[later] * gains[later]))
        rates[k] = math.log2(1.0 + P[k] * gains[k] / (interf + floor))
    return rates


def maxmin_feasible(
    t: float,
    gains: np.ndarray,
    floor: float,
    power_model: str,
    P_max: float | None = None,
    P_tot: float | None = None,
) -> bool:
    """Fixed-t max-min feasibility: can ALL K users reach rate t? Source: GT4 max-min
    fairness (CLAUDE.md Sec. 9).

    EXACTLY a QoS-feasibility check with the common floor t for every user
    (gamma(t) = 2^t - 1): REUSES ``power_lp`` as the oracle - no second solver. True
    iff a feasible power vector exists under the Version-A caps / Version-B budget.
    t <= 0 is trivially feasible (all rates are >= 0 at any P), returned without an
    LP call. MONOTONE in t (a lower common floor is always easier - relaxing every
    wall), which is what licenses ``maxmin_bisection``.
    """
    if t <= 0.0:
        return True
    _, _, status = power_lp(gains, floor, power_model, P_max=P_max, P_tot=P_tot, R_min=t)
    if status == "optimal":
        return True
    if status == "infeasible":
        return False
    raise RuntimeError(f"unexpected LP status {status!r} in maxmin_feasible")


def maxmin_bisection(
    gains: np.ndarray,
    floor: float,
    power_model: str,
    P_max: float | None = None,
    P_tot: float | None = None,
    t_tol: float = 1e-4,
) -> tuple[float, np.ndarray, int]:
    """Max-min fair power allocation via epigraph bisection. Source: GT4 max-min
    fairness (CLAUDE.md Sec. 9).

    Epigraph reformulation: max_t t s.t. R_{m,k} >= t for all k - the nested min is
    removed by the floor variable t; at the optimum t* = min_k R_{m,k}. Monotone
    feasibility (``maxmin_feasible``) => bisection on t is valid.

    Bracket: t_lo = 0 (always feasible); t_hi = log2(1 + max(g)*P_cap/floor) with
    P_cap = P_max (Version A) or P_tot (Version B) - the STRONGEST user's single-user
    full-power rate. No user's rate, hence no common floor, can exceed it (intra-segment
    interference and everyone else's power can only reduce a user's rate), so
    t* <= t_hi. Halve until t_hi - t_lo <= t_tol.

    Returns (t_star, P_star, n_iter): t_star = the last CERTIFIED-FEASIBLE floor
    (within t_tol of the true optimum), P_star = powers achieving it (one final LP
    call), n_iter = bisection count. Cost: n_iter = ceil(log2(t_hi/t_tol)) feasibility
    LPs + 1 final LP = O(log(1/t_tol)) (Gate F). Per-segment (GT4 separability).
    NOTE: max-min needs NO R_min - it MAXIMIZES the floor; R_min stays a QoS-only
    (P4-2) concept.
    """
    if power_model not in ("A", "B"):
        raise ValueError(f"power_model must be 'A' or 'B', got {power_model!r}")
    gains = np.asarray(gains, dtype=float)
    P_cap = P_max if power_model == "A" else P_tot
    if P_cap is None:
        raise ValueError(f"Version {power_model} requires "
                         f"{'P_max' if power_model == 'A' else 'P_tot'}")
    if t_tol <= 0.0:
        raise ValueError(f"t_tol must be > 0, got {t_tol}")

    t_lo, t_hi = 0.0, math.log2(1.0 + float(gains.max()) * float(P_cap) / floor)
    n_iter = 0
    while t_hi - t_lo > t_tol:
        mid = 0.5 * (t_lo + t_hi)
        if maxmin_feasible(mid, gains, floor, power_model, P_max=P_max, P_tot=P_tot):
            t_lo = mid
        else:
            t_hi = mid
        n_iter += 1
    P_star, _, status = power_lp(
        gains, floor, power_model, P_max=P_max, P_tot=P_tot,
        R_min=t_lo if t_lo > 0.0 else None)
    assert status == "optimal", f"final LP at certified-feasible t_lo failed: {status}"
    return t_lo, P_star, n_iter


def tdma_maxmin(
    gains: np.ndarray,
    floor: float,
    power_model: str,
    P_max: float | None = None,
    P_tot: float | None = None,
    t_tol: float = 1e-6,
) -> float:
    """Max-min floor of OPTIMALLY TIME-SHARED TDMA - the same boosted benchmark family
    as P4-2 Gate E - for the fairness crossover. Source: GT4 max-min fairness
    (CLAUDE.md Sec. 9); NOMA-vs-TDMA comparison per the two-sided honesty rule.

    User k is active alone a fraction tau_k of the slot (sum_k tau_k = 1),
    interference-free, at in-slot power Q_k; rate_k = tau_k*log2(1 + Q_k g_k/floor).

    Version "A" (per-user PEAK, headline) - CLOSED FORM: the in-slot power is capped at
    P_max (no boost beyond the peak), so Q_k = P_max (rate increasing in Q_k) and
    rate_k = tau_k*c_k with c_k = log2(1 + g_k P_max/floor) - LINEAR in tau. Max-min
    over the simplex equalizes tau_k*c_k = t: sum_k t/c_k = 1, hence
        t*_TDMA-A = 1 / sum_k (1/c_k)     (harmonic form; elementary derivation).

    Version "B" (shared AVERAGE budget) - SMALL NUMERIC SOLVE (documented, not closed
    form): in-slot boost Q_k free, average power sum_k tau_k Q_k <= P_tot. For a fixed
    floor t the cheapest slot power is Q_k = floor*(2^{t/tau_k}-1)/g_k, so feasibility
    is  min_{sum tau=1} sum_k (floor/g_k)*tau_k*(2^{t/tau_k}-1) <= P_tot - each term is
    CONVEX in tau_k, solved by SLSQP on the simplex; outer monotone bisection on t.
    Certified in tests against the SYMMETRIC closed check: equal gains g => tau_k = 1/K
    by symmetry+convexity => t*_TDMA-B = (1/K)*log2(1 + g*P_tot/floor).
    """
    if power_model not in ("A", "B"):
        raise ValueError(f"power_model must be 'A' or 'B', got {power_model!r}")
    gains = np.asarray(gains, dtype=float)
    if power_model == "A":
        if P_max is None:
            raise ValueError("Version A requires P_max")
        c = np.log2(1.0 + gains * float(P_max) / floor)
        return float(1.0 / np.sum(1.0 / c))

    if P_tot is None:
        raise ValueError("Version B requires P_tot")
    K = gains.size
    w = floor / gains

    def budget_needed(t: float) -> float:
        """min over the simplex of the average power required for common floor t."""
        def budget(tau: np.ndarray) -> float:
            s = np.clip(t / np.maximum(tau, 1e-12), 0.0, 500.0)   # overflow guard
            return float(np.sum(w * tau * (np.exp2(s) - 1.0)))
        if K == 1:
            return budget(np.array([1.0]))
        res = minimize(budget, np.full(K, 1.0 / K), method="SLSQP",
                       bounds=[(1e-6, 1.0)] * K,
                       constraints=[{"type": "eq",
                                     "fun": lambda tau: float(np.sum(tau) - 1.0)}])
        return float(res.fun)

    t_lo, t_hi = 0.0, math.log2(1.0 + float(gains.max()) * float(P_tot) / floor)
    while t_hi - t_lo > t_tol:
        mid = 0.5 * (t_lo + t_hi)
        if budget_needed(mid) <= float(P_tot):
            t_lo = mid
        else:
            t_hi = mid
    return t_lo


def segment_rate_from_powers(P: np.ndarray, gains: np.ndarray, floor: float) -> float:
    """TRUE per-segment sum rate at given powers: log2(1 + sum_k P_k g_k / floor).

    Source: GT4 true instantaneous rate (same formula as placement.rate_at_position,
    with the gains precomputed instead of derived from geometry) -- the cross-check
    quantity for the LP solution. Equals sum(per_user_rates) by the telescoping/SIC
    identity (asserted in tests).
    """
    P = np.asarray(P, dtype=float)
    gains = np.asarray(gains, dtype=float)
    if floor <= 0.0:
        raise ValueError(f"floor = I_m + N0 must be > 0, got {floor}")
    return math.log2(1.0 + float(P @ gains) / floor)
