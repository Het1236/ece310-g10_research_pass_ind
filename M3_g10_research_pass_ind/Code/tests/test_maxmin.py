"""Phase-4 P4-3 verification gates A-F for max-min fairness (src/power_alloc.py
maxmin_feasible / maxmin_bisection / tdma_maxmin). Sources: GT4 (CLAUDE.md Sec. 9,
max-min fairness).

Instances are stated via the per-user full-power SNRs x_k = g_k*P_max/floor (physical
gain scale g ~ 1e-8, floor ~ 2e-9 as in P4-2), so each regime is controlled exactly:
  * DISPARITY x = (36, 4.7): near/far users, the fairness-win regime;
  * SYMMETRIC x = (4, 4): equal channels, the honest-boundary regime.
Gate D is TWO-SIDED by construction: it asserts the NOMA win ONLY in the disparity
regime and RECORDS (never asserts) the symmetric outcome under both power models.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src import power_alloc

_FLOOR = 2.0e-9
_P_MAX = 1.0
_T_TOL = 1.0e-4


def _gains(x: list[float]) -> np.ndarray:
    """Gains realizing the given full-power SNRs x_k = g_k*P_max/floor."""
    return np.asarray(x, dtype=float) * _FLOOR / _P_MAX


G_DISP = _gains([36.0, 4.7])       # disparity (near/far)
G_SYM = _gains([4.0, 4.0])         # symmetric
G_K3 = _gains([36.0, 12.0, 4.7])   # 3-user for Gate A


def _grid_maxmin(gains: np.ndarray, floor: float, P_max: float, n: int) -> float:
    """Independent reference: exhaustive max-min floor over an n^K power grid,
    rates recomputed from first principles (fixed strongest-first SIC order)."""
    K = gains.size
    order = power_alloc.sic_order(gains)
    axes = [np.linspace(0.0, P_max, n)] * K
    Ps = np.meshgrid(*axes, indexing="ij")
    interf = np.zeros_like(Ps[0])
    rates = []
    for i in reversed(range(K)):            # decode-last user first: no interference
        k = int(order[i])
        rates.append(np.log2(1.0 + Ps[k] * gains[k] / (interf + floor)))
        interf = interf + Ps[k] * gains[k]
    return float(np.minimum.reduce(rates).max())


# ----------------------------------------------------------------------------
# GATE A: bisection <-> exhaustive power-grid max-min floor
# ----------------------------------------------------------------------------
def test_gateA_bisection_matches_grid_K2() -> None:
    """t* from bisection == exhaustive 501^2 power-grid floor (slack = grid step
    sensitivity ~ 5e-3; the grid can never exceed the true optimum by > t_tol)."""
    t_bis, _, _ = power_alloc.maxmin_bisection(
        G_DISP, _FLOOR, "A", P_max=_P_MAX, t_tol=_T_TOL)
    t_grid = _grid_maxmin(G_DISP, _FLOOR, _P_MAX, 501)
    assert t_grid <= t_bis + _T_TOL * 1.01, "grid beat the certified floor"
    assert abs(t_bis - t_grid) <= 5e-3, f"bisection {t_bis} vs grid {t_grid}"


def test_gateA_bisection_matches_grid_K3() -> None:
    """K=3: 81^3 grid (coarser => slack 3e-2, documented grid-resolution limit)."""
    t_bis, _, _ = power_alloc.maxmin_bisection(
        G_K3, _FLOOR, "A", P_max=_P_MAX, t_tol=_T_TOL)
    t_grid = _grid_maxmin(G_K3, _FLOOR, _P_MAX, 81)
    assert t_grid <= t_bis + _T_TOL * 1.01
    assert abs(t_bis - t_grid) <= 3e-2, f"bisection {t_bis} vs grid {t_grid}"


# ----------------------------------------------------------------------------
# GATE B: equalization at the optimum (both-tie and cap-limited cases)
# ----------------------------------------------------------------------------
def test_gateB_rates_equalize_at_optimum() -> None:
    """Both-tie case (x=(36,36): no user is cap-limited before the walls bind):
    at t* BOTH users' recomputed rates equal t* within ~5*t_tol - the max-min
    optimum lifts the floor until the users tie (GT4 epigraph prediction)."""
    gains = _gains([36.0, 36.0])
    t_star, P_star, _ = power_alloc.maxmin_bisection(
        gains, _FLOOR, "A", P_max=_P_MAX, t_tol=_T_TOL)
    rates = power_alloc.per_user_rates(P_star, gains, _FLOOR)
    assert np.all(np.abs(rates - t_star) <= 5 * _T_TOL), (
        f"rates {rates} not equalized to t*={t_star}")


def test_gateB_cap_limited_user_ties_floor_others_exceed() -> None:
    """Cap-limited case (disparity x=(36,4.7)): the weak user's peak caps the floor at
    its solo full-power rate log2(1+x2); it ties t* (at its cap), while the strong
    user sits strictly ABOVE t* - the documented exception to full equalization."""
    t_star, P_star, _ = power_alloc.maxmin_bisection(
        G_DISP, _FLOOR, "A", P_max=_P_MAX, t_tol=_T_TOL)
    assert t_star == pytest.approx(math.log2(1.0 + 4.7), abs=5 * _T_TOL)
    rates = power_alloc.per_user_rates(P_star, G_DISP, _FLOOR)
    k_weak = int(np.argmin(G_DISP))
    assert abs(rates[k_weak] - t_star) <= 5 * _T_TOL          # weak user ties the floor
    assert P_star[k_weak] == pytest.approx(_P_MAX, rel=1e-6)  # ... at its power cap
    assert rates.max() > t_star + 0.2                          # strong user exceeds it


# ----------------------------------------------------------------------------
# GATE C: monotone feasibility (the property licensing bisection)
# ----------------------------------------------------------------------------
def test_gateC_feasibility_monotone_in_t() -> None:
    t_star, _, _ = power_alloc.maxmin_bisection(
        G_DISP, _FLOOR, "A", P_max=_P_MAX, t_tol=_T_TOL)
    grid = np.linspace(0.0, 1.4 * t_star, 41)
    feas = np.array([power_alloc.maxmin_feasible(t, G_DISP, _FLOOR, "A", P_max=_P_MAX)
                     for t in grid])
    assert np.all(feas[:-1].astype(int) >= feas[1:].astype(int)), (
        "feasibility not monotone (True block then False block expected)")
    assert power_alloc.maxmin_feasible(t_star - 2 * _T_TOL, G_DISP, _FLOOR, "A",
                                       P_max=_P_MAX)
    assert not power_alloc.maxmin_feasible(t_star + 2 * _T_TOL, G_DISP, _FLOOR, "A",
                                           P_max=_P_MAX)


# ----------------------------------------------------------------------------
# GATE D: FAIRNESS-WIN, TWO-SIDED (mirror of P4-2 Gate E)
# ----------------------------------------------------------------------------
def test_gateD_disparity_noma_beats_tdma() -> None:
    """DISPARITY regime (x1/x2 = 36/4.7 near/far): NOMA's max-min floor strictly beats
    optimally time-shared TDMA under the headline Version-A peak model. NOMA serves the
    weak user continuously at its cap (floor log2(1+x2)=2.51) while TDMA must split
    time (harmonic floor 1/(1/c1+1/c2)=1.69) - the fairness win is REAL here."""
    t_noma, _, _ = power_alloc.maxmin_bisection(
        G_DISP, _FLOOR, "A", P_max=_P_MAX, t_tol=_T_TOL)
    t_tdma = power_alloc.tdma_maxmin(G_DISP, _FLOOR, "A", P_max=_P_MAX)
    assert t_noma > t_tdma + 0.5, f"NOMA {t_noma} vs TDMA {t_tdma}: no disparity win"


def test_gateD_symmetric_recorded_not_asserted() -> None:
    """SYMMETRIC regime (x=(4,4)): RECORD the outcome under BOTH power models; assert
    NO NOMA superiority. Under Version B the theory predicts an EXACT MAC tie
    (t = 0.5*log2(1+2x) both sides) - asserted as a TIE within numerics, which is an
    honest-boundary statement, not a win claim. Version A's outcome is logged only."""
    # Version A: record which side wins (peak-limited TDMA wastes silent users' power).
    t_noma_A, _, _ = power_alloc.maxmin_bisection(
        G_SYM, _FLOOR, "A", P_max=_P_MAX, t_tol=_T_TOL)
    t_tdma_A = power_alloc.tdma_maxmin(G_SYM, _FLOOR, "A", P_max=_P_MAX)
    # Version B (P_tot = K*P_max, confirmed): exact symmetric MAC tie expected.
    P_tot = 2 * _P_MAX
    t_noma_B, _, _ = power_alloc.maxmin_bisection(
        G_SYM, _FLOOR, "B", P_tot=P_tot, t_tol=_T_TOL)
    t_tdma_B = power_alloc.tdma_maxmin(G_SYM, _FLOOR, "B", P_tot=P_tot)
    print(f"\n[GATE D LOG] symmetric x=4: Version A NOMA {t_noma_A:.4f} vs TDMA "
          f"{t_tdma_A:.4f} ({'NOMA' if t_noma_A > t_tdma_A else 'TDMA'} ahead); "
          f"Version B NOMA {t_noma_B:.4f} vs TDMA {t_tdma_B:.4f} (tie expected)")
    x = 4.0
    t_tie = 0.5 * math.log2(1.0 + 2.0 * x)                    # symmetric MAC point
    assert abs(t_noma_B - t_tie) <= 1e-3
    assert abs(t_tdma_B - t_tie) <= 1e-3
    assert abs(t_noma_B - t_tdma_B) <= 2e-3, "symmetric Version-B should be a tie"


def test_gateD_tdma_B_certified_against_symmetric_closed_form() -> None:
    """Certifies tdma_maxmin's numeric Version-B path: equal gains => tau=1/K =>
    t* = (1/K)log2(1+g*P_tot/floor) (closed check from the docstring derivation);
    and B >= A at P_tot=K*P_max (boost + pooled budget can only help TDMA)."""
    g, P_tot = G_SYM, 2 * _P_MAX
    closed = 0.5 * math.log2(1.0 + float(g[0]) * P_tot / _FLOOR)
    assert power_alloc.tdma_maxmin(g, _FLOOR, "B", P_tot=P_tot) == pytest.approx(
        closed, abs=1e-4)
    t_A = power_alloc.tdma_maxmin(G_DISP, _FLOOR, "A", P_max=_P_MAX)
    t_B = power_alloc.tdma_maxmin(G_DISP, _FLOOR, "B", P_tot=2 * _P_MAX)
    assert t_B >= t_A - 1e-6, f"TDMA-B {t_B} < TDMA-A {t_A}: boost+pool should help"


# ----------------------------------------------------------------------------
# GATE E: K=1 sanity (no fairness tension)
# ----------------------------------------------------------------------------
def test_gateE_K1_maxmin_is_single_user_rate() -> None:
    gains = _gains([25.0])
    t_star, P_star, _ = power_alloc.maxmin_bisection(
        gains, _FLOOR, "A", P_max=_P_MAX, t_tol=_T_TOL)
    single = math.log2(1.0 + 25.0)
    assert abs(t_star - single) <= _T_TOL * 1.01
    assert P_star[0] == pytest.approx(_P_MAX, rel=1e-9)


# ----------------------------------------------------------------------------
# GATE F: bisection cost = O(log 1/t_tol) feasibility calls
# ----------------------------------------------------------------------------
def test_gateF_bisection_call_count(monkeypatch) -> None:
    """One LP per bisection iteration + 1 final solve; n_iter <=
    ceil(log2(t_hi/t_tol)) + 1 with t_hi the documented strongest-user bracket."""
    calls = 0
    real_lp = power_alloc.power_lp

    def counting_lp(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_lp(*args, **kwargs)

    monkeypatch.setattr(power_alloc, "power_lp", counting_lp)
    t_star, _, n_iter = power_alloc.maxmin_bisection(
        G_DISP, _FLOOR, "A", P_max=_P_MAX, t_tol=_T_TOL)
    t_hi = math.log2(1.0 + float(G_DISP.max()) * _P_MAX / _FLOOR)
    bound = math.ceil(math.log2(t_hi / _T_TOL)) + 1
    assert n_iter <= bound, f"n_iter={n_iter} exceeds O(log 1/t_tol) bound {bound}"
    assert calls == n_iter + 1, (
        f"{calls} LP calls for {n_iter} iterations (expected n_iter + 1 final)")
