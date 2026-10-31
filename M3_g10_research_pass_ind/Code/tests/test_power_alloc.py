"""Phase-4 P4-2 verification gates A-F for src/power_alloc.py. Sources: GT4 (CLAUDE.md
Sec. 9, power allocation).

Gate A certifies the LP against brute-force vertex enumeration on well-scaled synthetic
instances (the LP is pure math; certification does not need physical gains). Gates B/E
use PHYSICAL gains from a real drop + K-start placement so the solver is also exercised
at the ~1e-8 gain scale. Gate E is the REVIEWER-HONESTY INVARIANT: equal total power =>
NOMA sum rate == (boosted, optimally time-shared) TDMA sum rate; it exists to prevent a
false sum-rate-win claim from ever entering code or paper.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from src import analytic, channel, placement, power_alloc
from src.config import load_params


@pytest.fixture(scope="module")
def params() -> dict:
    return load_params()


@pytest.fixture(scope="module")
def phys(params) -> dict:
    p3, p4 = params["phase3"], params["phase4"]
    return {
        "eta": float(params["eta"]), "d": float(params["d"]),
        "D_S": float(params["D_S"]), "D_W": float(params["D_W"]),
        "P": float(p3["P"]), "N0": float(p3["N0"]),
        "P_max": float(p4["P_max"]), "R_min_grid": [float(r) for r in p4["R_min_grid"]],
        "tol": float(p4["gs_tol"]), "L1": analytic.L1_exact(params),
    }


def _physical_instance(K: int, seed: int, ph: dict) -> tuple[np.ndarray, float]:
    """Gains g_k = eta/((x_pa-x_k)^2 + y_k^2 + d^2) from a seeded drop with the PA
    placed by K-start golden-section (P4-1.5 production placement), plus the constant
    floor I_m + N0 = N0 + 2K*P*L1_exact (GT4 true rate)."""
    rng = np.random.default_rng(seed)
    ux = rng.uniform(0.0, ph["D_S"], K)
    uy = rng.uniform(-ph["D_W"] / 2.0, ph["D_W"] / 2.0, K)
    floor = ph["N0"] + 2.0 * K * ph["P"] * ph["L1"]

    def obj(x: float) -> float:
        return placement.rate_at_position(x, ux, uy, ph["eta"], ph["d"], ph["P"], floor)

    x_pa, _, _ = placement.golden_section_kstart(obj, 0.0, ph["D_S"], ux, ph["tol"])
    gains = channel.path_gain((x_pa - ux) ** 2 + uy ** 2 + ph["d"] ** 2, ph["eta"])
    return np.asarray(gains, dtype=float), floor


def _vertex_bruteforce(A_ub: np.ndarray, b_ub: np.ndarray, c: np.ndarray) -> float:
    """Reference LP solve: enumerate all K-subsets of active constraints (polytope
    vertex candidates), keep the feasible ones, return the best objective c@P. The LP
    optimum of a bounded feasible LP is attained at a vertex."""
    K = c.size
    best = -math.inf
    for rows in itertools.combinations(range(A_ub.shape[0]), K):
        A_sub, b_sub = A_ub[list(rows)], b_ub[list(rows)]
        if abs(np.linalg.det(A_sub)) < 1e-12:
            continue
        P_v = np.linalg.solve(A_sub, b_sub)
        if np.all(A_ub @ P_v <= b_ub + 1e-9):
            best = max(best, float(c @ P_v))
    return best


def _ub_form(gains, floor, power_model, P_max=None, P_tot=None, R_min=None):
    """Assemble the same constraint set power_lp uses, as A_ub @ P <= b_ub rows
    (bounds included as rows) for the vertex reference."""
    K = gains.size
    rows, vals = [-np.eye(K)], [np.zeros(K)]                    # P >= 0
    if power_model == "A":
        rows.append(np.eye(K)); vals.append(np.full(K, P_max))  # P <= P_max
    else:
        rows.append(np.ones((1, K))); vals.append(np.array([P_tot]))
    if R_min is not None and R_min > 0.0:
        A_qos, b_qos = power_alloc.build_qos_constraints(
            gains, power_alloc.sic_order(gains), power_alloc.qos_to_gamma(R_min), floor)
        rows.append(-A_qos); vals.append(-b_qos)
    return np.vstack(rows), np.concatenate(vals)


# ----------------------------------------------------------------------------
# Sanity: gamma map + telescoping identity
# ----------------------------------------------------------------------------
def test_qos_to_gamma() -> None:
    assert power_alloc.qos_to_gamma(0.0) == 0.0
    assert power_alloc.qos_to_gamma(1.0) == pytest.approx(1.0)
    assert power_alloc.qos_to_gamma(2.0) == pytest.approx(3.0)


def test_per_user_rates_telescope_to_sum_rate(phys) -> None:
    """sum_k R_k == log2(1 + sum P g / floor): the Phase-1 telescoping/SIC identity
    connects per_user_rates to segment_rate_from_powers."""
    gains, floor = _physical_instance(3, 101, phys)
    P = np.array([1.0, 0.6, 0.3])
    total = power_alloc.segment_rate_from_powers(P, gains, floor)
    assert float(np.sum(power_alloc.per_user_rates(P, gains, floor))) == pytest.approx(
        total, rel=1e-12)


def test_sic_order_is_decreasing_gain(phys) -> None:
    gains, _ = _physical_instance(4, 102, phys)
    order = power_alloc.sic_order(gains)
    assert np.all(np.diff(gains[order]) <= 0.0)


# ----------------------------------------------------------------------------
# GATE A: LP <-> brute-force vertex enumeration (tiny instances, both models)
# ----------------------------------------------------------------------------
# Synthetic O(1) instances: (gains, floor, R_min) chosen to cover non-binding QoS,
# BINDING QoS walls (all-caps infeasible for the wall, LP must trade off), and no-QoS.
_GATE_A_CASES = [
    # (model, gains, floor, R_min, P_max, P_tot)
    ("A", np.array([2.0, 1.5]), 0.4, None, 1.0, None),          # K=2, no QoS
    ("A", np.array([2.0, 1.5]), 0.4, math.log2(2.2), 1.0, None),# K=2, BINDING wall (gamma=1.2)
    ("A", np.array([3.0, 1.7, 0.9]), 0.5, 1.0, 1.0, None),      # K=3, binding wall-0
    ("A", np.array([3.0, 1.7, 0.9]), 0.4, 0.5, 1.0, None),      # K=3, feasible loose QoS
    ("B", np.array([3.0, 1.7, 0.9]), 0.4, None, None, 1.5),     # K=3 budget, no QoS
    ("B", np.array([3.0, 1.7, 0.9]), 0.4, math.log2(1.5), None, 1.5),  # K=3 budget + QoS
]


@pytest.mark.parametrize("model,gains,floor,R_min,P_max,P_tot", _GATE_A_CASES)
def test_gateA_lp_matches_vertex_bruteforce(model, gains, floor, R_min, P_max, P_tot) -> None:
    """LP optimum == best feasible polytope vertex (certifies the solver, GT4 LP item)."""
    P_star, obj, status = power_alloc.power_lp(
        gains, floor, model, P_max=P_max, P_tot=P_tot, R_min=R_min)
    assert status == "optimal"
    A_ub, b_ub = _ub_form(gains, floor, model, P_max=P_max, P_tot=P_tot, R_min=R_min)
    ref = _vertex_bruteforce(A_ub, b_ub, gains)
    assert obj == pytest.approx(ref, rel=1e-6), f"LP {obj} vs vertex reference {ref}"
    # the returned point is itself feasible
    assert np.all(A_ub @ P_star <= b_ub + 1e-9)


# ----------------------------------------------------------------------------
# GATE B: QoS satisfied at the LP solution (recomputed from powers, physical gains)
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("K", [2, 3, 4])
def test_gateB_qos_satisfied_at_solution(phys, K: int) -> None:
    """Every user's TRUE rate (recomputed via per_user_rates, not the LP's internal
    view) meets a feasible R_min from the config grid, under both power models."""
    gains, floor = _physical_instance(K, 200 + K, phys)
    R_min = phys["R_min_grid"][1]                       # 0.25 bits: robustly feasible
    for model, kw in (("A", {"P_max": phys["P_max"]}),
                      ("B", {"P_tot": K * phys["P_max"]})):
        P_star, _, status = power_alloc.power_lp(gains, floor, model, R_min=R_min, **kw)
        assert status == "optimal", f"model {model} unexpectedly {status}"
        rates = power_alloc.per_user_rates(P_star, gains, floor)
        assert np.all(rates >= R_min - 1e-9), f"model {model}: rates {rates} < {R_min}"


# ----------------------------------------------------------------------------
# GATE C: infeasibility detected cleanly
# ----------------------------------------------------------------------------
def test_gateC_infeasible_returned_cleanly(phys) -> None:
    """R_min far above the SNR ceiling: status 'infeasible', no crash, no silent junk."""
    gains, floor = _physical_instance(3, 300, phys)
    # ceiling: even alone at the cap, best user gets log2(1+g_max*P_max/floor) bits.
    ceiling = math.log2(1.0 + float(gains.max()) * phys["P_max"] / floor)
    P_star, obj, status = power_alloc.power_lp(
        gains, floor, "A", P_max=phys["P_max"], R_min=ceiling + 5.0)
    assert status == "infeasible" and P_star is None and math.isnan(obj)
    P_star, obj, status = power_alloc.power_lp(
        gains, floor, "B", P_tot=3 * phys["P_max"], R_min=ceiling + 5.0)
    assert status == "infeasible" and P_star is None and math.isnan(obj)


# ----------------------------------------------------------------------------
# GATE D: no-QoS corner predictions (GT4: objective increasing in every P_k)
# ----------------------------------------------------------------------------
def test_gateD_versionA_all_users_at_cap(phys) -> None:
    gains, floor = _physical_instance(3, 400, phys)
    P_star, _, status = power_alloc.power_lp(gains, floor, "A", P_max=phys["P_max"])
    assert status == "optimal"
    assert np.allclose(P_star, phys["P_max"], rtol=0.0, atol=1e-9)


def test_gateD_versionB_all_budget_on_best_user(phys) -> None:
    gains, floor = _physical_instance(3, 401, phys)
    P_tot = 3 * phys["P_max"]
    P_star, _, status = power_alloc.power_lp(gains, floor, "B", P_tot=P_tot)
    assert status == "optimal"
    k_best = int(np.argmax(gains))
    assert P_star[k_best] == pytest.approx(P_tot, rel=1e-9)
    mask = np.ones(3, dtype=bool); mask[k_best] = False
    assert np.all(np.abs(P_star[mask]) <= 1e-9 * P_tot)


# ----------------------------------------------------------------------------
# GATE E: REVIEWER-HONESTY INVARIANT (equal total power => NOMA == TDMA sum rate)
# ----------------------------------------------------------------------------
def test_gateE_equal_total_power_noma_tdma_sum_rate_tie(phys) -> None:
    """MAC sum-rate tie (GT4 ⚠ invariant): NOMA with powers {P_k} gives
    log2(1 + sum P_k g_k / floor); TDMA with the SAME average powers -- user k active a
    fraction tau_k = P_k g_k / sum_l P_l g_l of the slot at boosted in-slot power
    P_k/tau_k (equal per-user ENERGY, equal total power) -- gives
    sum_k tau_k log2(1 + (P_k/tau_k) g_k / floor), and every in-slot SINR equals the
    common value sum_l P_l g_l / floor, so the two sum rates are IDENTICAL. This test
    exists to prevent any sum-rate-win-over-TDMA claim at equal total power from ever
    entering the code or the paper."""
    gains, floor = _physical_instance(3, 500, phys)
    P = np.array([1.0, 0.7, 0.4])                     # arbitrary powers, same both sides
    noma = power_alloc.segment_rate_from_powers(P, gains, floor)
    collected = float(P @ gains)
    tau = P * gains / collected
    tdma = float(np.sum(tau * np.log2(1.0 + (P / tau) * gains / floor)))
    assert abs(noma - tdma) <= 1e-9, f"NOMA {noma} vs boosted time-shared TDMA {tdma}"
    # and the LP optimum obeys the same tie (Version B at the same total power)
    P_star, _, status = power_alloc.power_lp(gains, floor, "B", P_tot=float(P.sum()))
    assert status == "optimal"
    noma_lp = power_alloc.segment_rate_from_powers(P_star, gains, floor)
    coll = float(P_star @ gains)
    act = P_star > 0.0                                # inactive users: tau=0, no slot
    tau = P_star[act] * gains[act] / coll
    tdma_lp = float(np.sum(tau * np.log2(1.0 + (P_star[act] / tau) * gains[act] / floor)))
    assert abs(noma_lp - tdma_lp) <= 1e-9


# ----------------------------------------------------------------------------
# GATE F: K=1 EDMA-collapse continuity
# ----------------------------------------------------------------------------
def test_gateF_K1_transmit_at_cap(phys) -> None:
    """K=1 Version A: the LP reduces to 'transmit at the peak' (single increasing
    objective, no intra-segment interference)."""
    gains, floor = _physical_instance(1, 600, phys)
    P_star, obj, status = power_alloc.power_lp(gains, floor, "A", P_max=phys["P_max"])
    assert status == "optimal"
    assert P_star[0] == pytest.approx(phys["P_max"], rel=1e-12)
    assert obj == pytest.approx(phys["P_max"] * gains[0], rel=1e-12)
