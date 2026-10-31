"""Gates for src/maxmin_closed.py (closed-form max-min allocation + proven Lipschitz
bound; V4 plan items A3 + F-2).

Gate A  equivalence  : maxmin_closed == power_alloc.maxmin_bisection within the
                       bisection's own t_tol, over a random battery, both power models.
Gate B  telescoping  : sum_k R_k == K * t* at the closed-form allocation.
Gate C  minimality   : no feasible allocation achieving all rates >= t uses less power in
                       any coordinate (checked against a dense grid at K = 2).
Gate D  Lipschitz    : lipschitz_V >= the max finite-difference slope of V on a dense
                       grid. THIS GATE IS WHAT MAKES PROPOSITION 6 TRUE.
Gate E  reference    : the two pinned instances of tests/test_maxmin_global.py still give
                       V* = 2.579661 and V* = 2.77412 through the fast path.
Gate F  determinism  : repeated calls return bit-identical results.

``power_alloc`` is NOT modified by this work; it is the reference oracle for Gate A.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src import analytic, maxmin_closed, maxmin_global, power_alloc
from src.config import load_params

_T_TOL = 1e-6            # bisection tolerance used for the oracle in Gate A
_RNG_SEED = 20260815


@pytest.fixture(scope="module")
def params() -> dict:
    return load_params()


@pytest.fixture(scope="module")
def env(params) -> dict:
    """Physical environment matching tests/test_maxmin_global.py."""
    K = 2
    floor = (float(params["phase3"]["N0"])
             + 2.0 * K * float(params["phase3"]["P"]) * analytic.L1_exact(params))
    return {
        "D_S": float(params["D_S"]), "d": float(params["d"]),
        "eta": float(params["eta"]), "P_max": float(params["phase4"]["P_max"]),
        "floor": floor,
    }


def _random_instance(rng, K: int, env: dict) -> np.ndarray:
    """Physically-scaled gains: users dropped in a segment, gains eta/((x-x_k)^2+C_k)."""
    ux = rng.uniform(0.0, env["D_S"], K)
    uy = rng.uniform(-5.0, 5.0, K)
    x = rng.uniform(0.0, env["D_S"])
    return env["eta"] / ((x - ux) ** 2 + uy ** 2 + env["d"] ** 2)


# --------------------------------------------------------------------------- Gate A
@pytest.mark.parametrize("power_model", ["A", "B"])
@pytest.mark.parametrize("K", [2, 3, 4, 6, 8])
def test_gateA_matches_bisection_oracle(K, power_model, env):
    """Closed form and the LP-bisection oracle agree to within the bisection's own t_tol.

    The two routes must solve the same problem, so |t_closed - t_oracle| <= t_tol is the
    headline assertion.

    ON THE SIGN OF THE RESIDUAL. ``maxmin_bisection`` returns its last certified-FEASIBLE
    iterate, so it should sit at or below the exact optimum. It occasionally sits ABOVE
    it, by ~1e-7, on Version-A instances. That is NOT a defect in the closed form: at such
    a t the componentwise-minimal powers exceed the per-user cap, so no feasible
    allocation attains that floor, and the LP accepted it only within HiGHS's own primal
    feasibility tolerance (recomputing the rates from the LP's returned powers shows a
    shortfall of the same order). ``test_gateA_disagreements_favour_the_closed_form``
    below asserts exactly that, so the direction of every disagreement is accounted for
    rather than absorbed into a loose tolerance.
    """
    rng = np.random.default_rng(_RNG_SEED + K + (0 if power_model == "A" else 1000))
    kw = ({"P_max": env["P_max"]} if power_model == "A"
          else {"P_tot": K * env["P_max"]})
    for _ in range(12):
        g = _random_instance(rng, K, env)
        t_ref, _, _ = power_alloc.maxmin_bisection(
            g, env["floor"], power_model, t_tol=_T_TOL, **kw)
        t_cf, _ = maxmin_closed.maxmin_closed(g, env["floor"], power_model, **kw)
        assert abs(t_cf - t_ref) <= _T_TOL + 1e-9, (
            f"closed form {t_cf} and oracle {t_ref} differ by more than t_tol")


@pytest.mark.parametrize("K", [2, 4, 6])
def test_gateA_disagreements_favour_the_closed_form(K, env):
    """Wherever the oracle reports a HIGHER floor than the closed form, that floor is
    genuinely unattainable: the componentwise-minimal powers for it exceed the cap.

    This pins the direction of the residual in Gate A. Without it, a real error in the
    recursion could hide inside the t_tol band.
    """
    rng = np.random.default_rng(_RNG_SEED + K)
    checked = 0
    for _ in range(12):
        g = _random_instance(rng, K, env)
        t_ref, _, _ = power_alloc.maxmin_bisection(
            g, env["floor"], "A", P_max=env["P_max"], t_tol=_T_TOL)
        t_cf, _ = maxmin_closed.maxmin_closed(g, env["floor"], "A", P_max=env["P_max"])
        if t_ref <= t_cf:
            continue
        checked += 1
        P_at_ref = maxmin_closed.minpower_sic(2.0 ** t_ref - 1.0, g, env["floor"])
        assert P_at_ref.max() > env["P_max"], (
            f"oracle floor {t_ref} exceeds the closed form {t_cf} but its minimal powers "
            f"fit inside the cap: the closed form would then be wrong")
    if checked == 0:
        pytest.skip("no disagreement of this sign in this battery")


@pytest.mark.parametrize("power_model", ["A", "B"])
@pytest.mark.parametrize("K", [2, 3, 4])
def test_gateA_returned_powers_are_feasible_and_achieve_the_floor(K, power_model, env):
    """The returned powers respect their constraint and every user's RECOMPUTED rate
    meets t*. Rates are recomputed from the powers by ``power_alloc.per_user_rates``
    rather than trusted from the solver."""
    rng = np.random.default_rng(_RNG_SEED + 7 * K)
    P_cap = env["P_max"] if power_model == "A" else K * env["P_max"]
    kw = {"P_max": P_cap} if power_model == "A" else {"P_tot": P_cap}
    for _ in range(12):
        g = _random_instance(rng, K, env)
        t_star, P = maxmin_closed.maxmin_closed(g, env["floor"], power_model, **kw)
        assert np.all(P >= 0.0)
        if power_model == "A":
            assert np.all(P <= P_cap * (1.0 + 1e-9)), "a power exceeds the peak cap"
            assert np.isclose(P.max(), P_cap, rtol=1e-9), (
                "at the Version-A optimum the binding user must sit exactly at the cap")
        else:
            assert P.sum() <= P_cap * (1.0 + 1e-9), "budget exceeded"
            assert np.isclose(P.sum(), P_cap, rtol=1e-9), (
                "at the Version-B optimum the whole budget must be spent")
        rates = power_alloc.per_user_rates(P, g, env["floor"])
        assert np.all(rates >= t_star - 1e-9), "a user misses the reported floor"


# --------------------------------------------------------------------------- Gate B
@pytest.mark.parametrize("K", [2, 3, 5])
def test_gateB_telescoping_identity(K, env):
    """sum_k R_k == K*t* at the closed-form allocation, i.e. every user sits exactly on
    the floor. This is the SIC telescoping identity specialised to the equal-rate point,
    and it is the cheapest available check that the backward recursion is right."""
    rng = np.random.default_rng(_RNG_SEED + 31 * K)
    for _ in range(12):
        g = _random_instance(rng, K, env)
        t_star, P = maxmin_closed.maxmin_closed(
            g, env["floor"], "B", P_tot=K * env["P_max"])
        rates = power_alloc.per_user_rates(P, g, env["floor"])
        assert np.allclose(rates, t_star, rtol=0, atol=1e-9), (
            f"rates {rates} are not all equal to t* = {t_star}")
        total = power_alloc.segment_rate_from_powers(P, g, env["floor"])
        assert math.isclose(total, K * t_star, rel_tol=1e-9), (
            f"telescoping identity broken: {total} != {K} * {t_star}")


# --------------------------------------------------------------------------- Gate C
def test_gateC_componentwise_minimality_K2(env):
    """``minpower_sic`` is componentwise minimal: shrinking ANY coordinate breaks the
    rate floor. Checked at K = 2 by scaling each coordinate down on a fine grid."""
    rng = np.random.default_rng(_RNG_SEED + 99)
    for _ in range(8):
        g = _random_instance(rng, 2, env)
        t = 0.8
        gamma = 2.0 ** t - 1.0
        P = maxmin_closed.minpower_sic(gamma, g, env["floor"])
        assert np.all(power_alloc.per_user_rates(P, g, env["floor"]) >= t - 1e-12)
        for k in range(2):
            for shrink in (0.999, 0.99, 0.9):
                Q = P.copy()
                Q[k] *= shrink
                rates = power_alloc.per_user_rates(Q, g, env["floor"])
                assert rates.min() < t - 1e-15, (
                    f"shrinking user {k} by {shrink} kept the floor: not minimal")


# --------------------------------------------------------------------------- Gate D
@pytest.mark.parametrize("power_model", ["A", "B"])
@pytest.mark.parametrize("K", [2, 3, 4])
def test_gateD_lipschitz_bound_is_valid(K, power_model, env):
    """THE GATE THAT MAKES PROPOSITION 6 TRUE.

    ``lipschitz_V`` must be at least the largest finite-difference slope of V on a dense
    grid across the whole segment (the grid crosses the SIC breakpoints, so this is the
    strictest form of the check: the bound must dominate every cell at once).
    """
    rng = np.random.default_rng(_RNG_SEED + 13 * K + (0 if power_model == "A" else 500))
    P_cap = env["P_max"] if power_model == "A" else K * env["P_max"]
    kw = {"P_max": P_cap} if power_model == "A" else {"P_tot": P_cap}
    xs = np.linspace(0.0, env["D_S"], 2000)
    for _ in range(10):
        ux = rng.uniform(0.0, env["D_S"], K)
        uy = rng.uniform(-5.0, 5.0, K)
        C = uy ** 2 + env["d"] ** 2
        vs = np.array([
            maxmin_closed.maxmin_closed(
                env["eta"] / ((x - ux) ** 2 + C), env["floor"], power_model, **kw)[0]
            for x in xs])
        observed = float(np.max(np.abs(np.diff(vs)) / np.diff(xs)))
        L = maxmin_closed.lipschitz_V(uy, env["d"])
        assert L >= observed, (
            f"Lipschitz bound {L} violated by observed slope {observed}")
        # The universal form must also hold and must dominate the per-instance form.
        assert 1.0 / (env["d"] * math.log(2.0)) >= L - 1e-12


def test_gateD_universal_bound_is_geometry_only(env):
    """L depends only on the user offsets and the height, not on K, powers, or the
    floor. Stated in Proposition 6(ii), so it is asserted rather than assumed."""
    uy = np.array([0.3, 4.6])
    base = maxmin_closed.lipschitz_V(uy, env["d"])
    assert math.isclose(base, 1.0 / (math.log(2.0) * math.sqrt(0.3 ** 2 + env["d"] ** 2)))
    assert base <= 1.0 / (env["d"] * math.log(2.0))
    # adding users further off axis cannot lower the bound (min over sqrt(C_k))
    assert maxmin_closed.lipschitz_V(np.append(uy, 9.0), env["d"]) == base


# --------------------------------------------------------------------------- Gate E
@pytest.mark.parametrize(
    "ux, uy, V_ref, x_ref",
    [
        ([12.416, 10.171], [-1.790, -3.107], 2.579661, 12.415),
        ([4.0, 9.0], [0.3, 4.6], 2.77412, 3.999),
    ],
    ids=["pinned", "disparity"],
)
def test_gateE_reference_instances_through_fast_path(ux, uy, V_ref, x_ref, env):
    """The two committed reference instances still certify to the same optimum when the
    value function is the closed form and the Lipschitz constant is the proven bound."""
    ux, uy = np.asarray(ux), np.asarray(uy)
    C = uy ** 2 + env["d"] ** 2

    def V(x: float) -> float:
        gains = env["eta"] / ((x - ux) ** 2 + C)
        return maxmin_closed.maxmin_closed(
            gains, env["floor"], "A", P_max=env["P_max"])[0]

    L = maxmin_closed.lipschitz_V(uy, env["d"])
    bps = maxmin_global.sic_breakpoints(ux, uy, env["d"], 0.0, env["D_S"])
    edges = [0.0] + bps + [env["D_S"]]
    best_x, best_f, worst_gap = None, -np.inf, 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < 1e-9:
            continue
        xc, fc, uc, _ = maxmin_global.piyavskii_max(V, a, b, L, tol=1e-3)
        if fc > best_f:
            best_x, best_f = xc, fc
        worst_gap = max(worst_gap, uc - best_f)
    assert worst_gap <= 1e-3 + 1e-9, f"certified gap {worst_gap} exceeds tolerance"
    assert abs(best_f - V_ref) <= 5e-3, f"V* {best_f} != reference {V_ref}"
    assert abs(best_x - x_ref) <= 0.05, f"x* {best_x} != reference {x_ref}"


# --------------------------------------------------------------------------- Gate F
def test_gateF_determinism(env):
    """Repeated calls are bit-identical: no random initialisation anywhere."""
    g = _random_instance(np.random.default_rng(_RNG_SEED), 3, env)
    vals = [maxmin_closed.maxmin_closed(g, env["floor"], "A", P_max=env["P_max"])[0]
            for _ in range(25)]
    assert max(vals) - min(vals) == 0.0, f"non-deterministic: spread {max(vals)-min(vals)}"


def test_gateF_K1_reduces_to_solo_rate_at_cap(env):
    """K = 1 collapse: the single user transmits at its cap and the floor is its solo
    rate. Mirrors ``power_alloc`` Gate F, which asserts the same for the LP route."""
    g = np.array([3.0e-8])
    t_star, P = maxmin_closed.maxmin_closed(g, env["floor"], "A", P_max=env["P_max"])
    assert math.isclose(float(P[0]), env["P_max"], rel_tol=1e-9)
    assert math.isclose(
        t_star, math.log2(1.0 + g[0] * env["P_max"] / env["floor"]), rel_tol=1e-9)
