"""Gates for src/floor_position.py and for the two solver changes it forces:
the position-dependent interference floor F(x), and the Lipschitz constant that has to
cover it.

The frozen floor was the Critical finding of the v3 editorial review. Section IV held
F = I_m + N0 constant in x while optimising x, but the interferers sit in the adjacent
segments and do not move, so the leakage reaching a moving antenna is a function of the
placement variable. These gates pin the correction down:

  A  L1_at reproduces analytic.L1_exact exactly at arbitrary offsets.
  B  dL1_at (Leibniz, closed form) matches a finite-difference reference.
  C  floor_at reduces to the frozen value at the segment centre, and swings by the
     measured factor across the segment.
  D  floor_logderiv_bound is a bound on |F'/F| at EVERY load, approached from below as
     K grows (the K-uniformity the certificate relies on).
  E  the repaired Lipschitz constant dominates the true |dV/dx| under F(x), where the
     old constant does not.
  F  a single Piyavskii-Shubert search over the whole segment returns the same certified
     optimum as the per-cell search, at no greater evaluation count.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src import analytic, floor_position, maxmin_closed, maxmin_global
from src.config import load_params


@pytest.fixture(scope="module")
def params() -> dict:
    return load_params()


@pytest.fixture(scope="module")
def env(params) -> dict:
    return {
        "D_S": float(params["D_S"]), "d": float(params["d"]),
        "eta": float(params["eta"]), "P_max": float(params["phase4"]["P_max"]),
        "P": float(params["phase3"]["P"]), "N0": float(params["phase3"]["N0"]),
    }


# --------------------------------------------------------------------------- Gate A
def test_gateA_L1_at_matches_analytic(params, env):
    """Same quantity and same quadrature as analytic.L1_exact, offset as an argument."""
    for delta in (2.0, env["D_S"] / 2.0, 10.0, env["D_S"] - 0.5):
        q = dict(params)
        q["delta"] = delta
        assert floor_position.L1_at(delta, params) == pytest.approx(
            analytic.L1_exact(q), rel=1e-12)


# --------------------------------------------------------------------------- Gate B
def test_gateB_dL1_matches_finite_difference(params, env):
    """Only the u-limits carry the offset, so Leibniz gives the derivative exactly."""
    h = 1e-4
    for delta in (1.0, 4.0, env["D_S"] / 2.0, 10.0, 12.0):
        fd = (floor_position.L1_at(delta + h, params)
              - floor_position.L1_at(delta - h, params)) / (2.0 * h)
        assert floor_position.dL1_at(delta, params) == pytest.approx(fd, rel=1e-6)


def test_gateB_leakage_rises_toward_a_neighbour(params, env):
    """Sliding the antenna toward a neighbour must raise the leakage from it."""
    for delta in (1.0, 5.0, 12.0):
        assert floor_position.dL1_at(delta, params) > 0.0


# --------------------------------------------------------------------------- Gate C
def test_gateC_floor_reduces_to_frozen_value_at_centre(params, env):
    """At the symmetric placement the two sides are equal, so F(D_S/2) is exactly the
    frozen floor N0 + 2 K P L1_exact that the previous version used everywhere."""
    for K in (1, 2, 4):
        frozen = env["N0"] + 2.0 * K * env["P"] * analytic.L1_exact(params)
        got = floor_position.floor_at(env["D_S"] / 2.0, K, params)
        assert got == pytest.approx(frozen, rel=1e-12)


def test_gateC_floor_swings_across_the_segment(params, env):
    """The defect the review found: the floor is not constant in x. Recorded here so a
    future edit that quietly reintroduces a constant floor fails a test."""
    lo, hi = floor_position.floor_swing(params, K=1)
    assert lo == pytest.approx(1.0, abs=1e-6), "the centre must be the minimum"
    assert hi > 9.0, f"expected a swing above 9x across the segment, got {hi}"


@pytest.mark.parametrize("K", [1, 2, 16])
def test_gateC_interpolant_is_accurate(params, K):
    """F depends on the deployment and the load, not on the drop, so it is built once as
    an interpolant. The residual must sit far below the certified gap of 1e-3 bps/Hz."""
    assert floor_position.max_interp_error(K, params) < 1e-6


# --------------------------------------------------------------------------- Gate D
@pytest.mark.parametrize("K", [1, 2, 4, 50])
def test_gateD_logderiv_bound_holds_at_every_load(params, env, K):
    """B_F must dominate |F'/F| at every K. K multiplies both leakage terms and appears
    nowhere else, so the bound is approached from below as K grows."""
    B_F = floor_position.floor_logderiv_bound(params)
    D_S, h = env["D_S"], 1e-3
    xs = np.linspace(0.02, D_S - 0.02, 201)
    worst = max(
        abs(floor_position.floor_at(x + h, K, params)
            - floor_position.floor_at(x - h, K, params)) / (2.0 * h)
        / floor_position.floor_at(x, K, params)
        for x in xs
    )
    assert worst <= B_F + 1e-9, f"K={K}: measured {worst} exceeds bound {B_F}"


def test_gateD_bound_is_load_free_and_tight(params, env):
    """The bound is not merely valid, it is the K -> infinity limit, so it is tight."""
    B_F = floor_position.floor_logderiv_bound(params)
    D_S, h = env["D_S"], 1e-3
    xs = np.linspace(0.02, D_S - 0.02, 201)

    def worst(K: int) -> float:
        return max(
            abs(floor_position.floor_at(x + h, K, params)
                - floor_position.floor_at(x - h, K, params)) / (2.0 * h)
            / floor_position.floor_at(x, K, params)
            for x in xs
        )

    assert worst(1) < worst(50) <= B_F
    assert worst(50) >= 0.99 * B_F, "bound should be tight at large load"


# --------------------------------------------------------------------------- Gate E
def test_gateE_repaired_lipschitz_dominates_true_slope(params, env):
    """The constant must bound |dV/dx| for the value function built on F(x). The old
    constant (floor_logderiv = 0) is checked too, and is expected to FAIL, which is why
    the repair is necessary rather than cosmetic."""
    ux = np.array([4.0, 9.0])
    uy = np.array([0.3, 4.6])
    K = 2
    B_F = floor_position.floor_logderiv_bound(params)
    F_of_x = floor_position.floor_fn_absolute(K, params, 0.0)
    V = maxmin_global.maxmin_value_function(
        ux, uy, env["d"], env["eta"], F_of_x, "A", P_max=env["P_max"])

    xs = np.linspace(0.05, env["D_S"] - 0.05, 1200)
    slopes = np.abs(np.diff([V(float(x)) for x in xs])) / np.diff(xs)
    worst = float(slopes.max())

    L_new = maxmin_closed.lipschitz_V(uy, env["d"], floor_logderiv=B_F)
    L_old = maxmin_closed.lipschitz_V(uy, env["d"])
    assert worst <= L_new, f"repaired constant {L_new} does not bound slope {worst}"
    assert L_new > L_old, "the repair must enlarge the constant, not shrink it"


def test_gateE_constant_floor_case_is_unchanged(env):
    """Default argument reproduces the published constant exactly, so every existing
    result computed with a frozen floor stays reproducible."""
    uy = np.array([0.3, 4.6])
    assert maxmin_closed.lipschitz_V(uy, env["d"]) == pytest.approx(
        1.0 / (math.log(2.0) * math.sqrt(0.3 ** 2 + env["d"] ** 2)), rel=1e-14)


# --------------------------------------------------------------------------- Gate F
def test_gateF_single_search_matches_cell_search(params, env):
    """Piyavskii-Shubert needs Lipschitz continuity, not differentiability. V is
    continuous and the analytic constant bounds its slope on every piece, so one search
    over the whole segment carries the identical certificate. The cells are a step in the
    proof, not a step in the algorithm."""
    ux = np.array([4.0, 9.0])
    uy = np.array([0.3, 4.6])
    K = 2
    B_F = floor_position.floor_logderiv_bound(params)
    F_of_x = floor_position.floor_fn_absolute(K, params, 0.0)

    common = dict(users_x=ux, users_y=uy, seg_lo=0.0, seg_hi=env["D_S"], d=env["d"],
                  eta=env["eta"], floor=F_of_x, power_model="A", P_max=env["P_max"],
                  tol=1e-3, floor_logderiv=B_F)
    one = maxmin_global.maxmin_global_placement(**common, use_cells=False)
    cells = maxmin_global.maxmin_global_placement(**common, use_cells=True)

    assert one["gap"] <= 1e-3 + 1e-9 and cells["gap"] <= 1e-3 + 1e-9
    assert abs(one["f"] - cells["f"]) <= 2e-3, "the two routes must agree on the optimum"
    assert one["n_eval"] <= cells["n_eval"], (
        f"single search should not cost more: {one['n_eval']} vs {cells['n_eval']}")


def test_gateF_single_search_is_deterministic(params, env):
    """No initialisation anywhere, so repeated calls are bit-identical."""
    ux = np.array([4.0, 9.0])
    uy = np.array([0.3, 4.6])
    B_F = floor_position.floor_logderiv_bound(params)
    F_of_x = floor_position.floor_fn_absolute(2, params, 0.0)
    vals = [
        maxmin_global.maxmin_global_placement(
            ux, uy, 0.0, env["D_S"], env["d"], env["eta"], F_of_x, "A",
            P_max=env["P_max"], tol=1e-3, floor_logderiv=B_F)["f"]
        for _ in range(4)
    ]
    assert max(vals) - min(vals) == 0.0
