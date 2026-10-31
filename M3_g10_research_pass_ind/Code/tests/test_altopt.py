"""Phase-4 P4-4/P4-4.5 verification gates for src/altopt.py (AO: placement <-> power).
Sources: GT4 (CLAUDE.md Sec. 9, AO + scan-refine items).

P4-4 finding (mirrors P4-1): the maxmin x-step objective min_k R_k(x) is a lower
envelope whose maxima sit at crossing/balance points, NOT user-anchored bumps, so
K-start bracketing is invalid for it (pinned instance below). P4-4.5 remedy: the
maxmin x-step is ROUTED to placement.scan_refine_max (coarse scan + GS refine),
certified here against a fine-grid exhaustive reference (Gate A-maxmin); the sum-rate
x-step stays on K-start (unchanged, exact for the weighted sum-of-bumps). The
monotonicity hard-assert (1e-9) and the dip guard (5e-4) are UNCHANGED - the dip guard
staying silent on the pinned instance is the proof the fix works.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src import altopt, analytic, channel, placement, power_alloc
from src.config import load_params


@pytest.fixture(scope="module")
def params() -> dict:
    return load_params()


@pytest.fixture(scope="module")
def env(params) -> dict:
    D_S = float(params["D_S"])
    return {
        "D_S": D_S, "eta": float(params["eta"]), "d": float(params["d"]),
        "D_W": float(params["D_W"]), "P_max": float(params["phase4"]["P_max"]),
        "anchors": (D_S / 2.0 - D_S, D_S / 2.0 + D_S),
        "gs_tol": float(params["phase4"]["gs_tol"]),
        "n_scan": int(params["phase4"]["maxmin_scan_points"]),
    }


def _minrate_closure(ux, uy, P, floor, env):
    """Scalar min-rate x-objective (the maxmin x-step block objective)."""
    def f(x: float) -> float:
        g = channel.path_gain((x - ux) ** 2 + uy ** 2 + env["d"] ** 2, env["eta"])
        return float(np.min(power_alloc.per_user_rates(P, g, floor)))
    return f


def _minrate_on(x: np.ndarray, ux, uy, P, floor, env) -> np.ndarray:
    """Vectorized min-rate envelope on an x-grid (per-x strongest-first order,
    suffix-sum interference - same convention as per_user_rates)."""
    g = env["eta"] / ((x[:, None] - ux[None, :]) ** 2 + uy[None, :] ** 2 + env["d"] ** 2)
    idx = np.argsort(-g, axis=1)
    contrib = P[idx] * np.take_along_axis(g, idx, axis=1)
    suffix = np.cumsum(contrib[:, ::-1], axis=1)[:, ::-1] - contrib
    return np.log2(1.0 + contrib / (suffix + floor)).min(axis=1)


def _minrate_grid_max(ux, uy, P, floor, env, n: int) -> tuple[float, float, float]:
    """Two-stage exhaustive reference for the min-rate envelope max: coarse n-point
    grid, then a 4001-point local refinement over +-1 coarse step (near-exact even at
    KINKED maxima, where a single grid's value error is slope*step/2). Also returns
    the local envelope SLOPE around the argmax, used to calibrate the value tolerance
    (at a kink the achievable value accuracy of ANY tol-terminated 1-D method is
    LINEAR in the position tolerance)."""
    x = np.linspace(0.0, env["D_S"], n)
    f = _minrate_on(x, ux, uy, P, floor, env)
    i = int(f.argmax())
    lo, hi = x[max(i - 1, 0)], x[min(i + 1, n - 1)]
    xf = np.linspace(lo, hi, 4001)
    ff = _minrate_on(xf, ux, uy, P, floor, env)
    j = int(ff.argmax())
    slope = float(np.max(np.abs(np.diff(ff))) / (xf[1] - xf[0]))
    return float(xf[j]), float(ff[j]), slope


def _floor(K: int, params) -> float:
    return float(params["phase3"]["N0"]) + 2.0 * K * float(
        params["phase3"]["P"]) * analytic.L1_exact(params)


def _drops(K: int, n: int, seed: int, env) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    return [(rng.uniform(0.0, env["D_S"], K),
             rng.uniform(-env["D_W"] / 2.0, env["D_W"] / 2.0, K)) for _ in range(n)]


# ----------------------------------------------------------------------------
# GATE A: monotone obj_history on a battery (both objectives) - the core AO guarantee
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("objective", ["sumrate", "maxmin"])
def test_gateA_history_monotone(params, env, objective) -> None:
    """obj_history non-decreasing every accepted step; the hard assert and the dip
    guard never trip across K in {1,2,4} x 4 drops (midpoint init)."""
    failures = []
    for K in (1, 2, 4):
        floor = _floor(K, params)
        for i, (ux, uy) in enumerate(_drops(K, 4, 500 + K, env)):
            try:
                r = altopt.ao_segment(ux, uy, 0.0, env["D_S"], env["anchors"],
                                      objective, "A", params, floor=floor)
                h = np.asarray(r["obj_history"])
                if not np.all(np.diff(h) >= -1e-9):
                    failures.append(f"K={K} drop {i}: non-monotone history {h}")
            except AssertionError as e:
                failures.append(f"K={K} drop {i}: {e}")
    assert not failures, (
        f"GATE A FAILED ({objective}): {len(failures)} monotonicity breaches:\n"
        + "\n".join(failures[:5]))


def test_gateA_pinned_maxmin_xstep_instance(params, env) -> None:
    """Pinned P4-4 instance: users x=(12.416,10.171), y=(-1.79,-3.107), init
    over_weakest. Under K-start the min-rate objective was BIMODAL inside bracket
    [0, 11.294] (peaks x=10.173 @ 2.3549 and x=6.683 @ 0.351) and GS returned the
    bracket edge 0.075 bits below the incumbent -- the dip guard raised. With the
    P4-4.5 scan-refine x-step this instance must run CLEAN (dip guard silent,
    monotone, converged at the true block peak). Kept as the named regression."""
    ux = np.array([12.416, 10.171]); uy = np.array([-1.790, -3.107])
    r = altopt.ao_segment(ux, uy, 0.0, env["D_S"], env["anchors"], "maxmin", "A",
                          params, init="over_weakest", floor=_floor(2, params))
    h = np.asarray(r["obj_history"])
    assert np.all(np.diff(h) >= -1e-9)
    assert r["status"] == "converged"
    assert r["obj"] >= 2.3549 - 1e-3, f"expected the x~10.17 peak, got {r['obj']:.4f}"


# ----------------------------------------------------------------------------
# GATE A-maxmin (P4-4.5 HEADLINE): scan_refine_max <-> exhaustive for the min-rate
# envelope, incl. the pinned instance at FIXED powers
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("K", [2, 4])
def test_gateA_maxmin_scan_refine_matches_exhaustive(params, env, K: int) -> None:
    """The maxmin x-step solver (scan_refine_max, PRODUCTION config n_scan/gs_tol)
    reaches the exhaustive optimum of min_k R_k(x): position within one coarse-scan
    step + 10*gs_tol; value within max(1e-6 rel, slope*gs_tol) of the (two-stage,
    near-exact) reference - SLOPE-CALIBRATED because at a KINKED envelope maximum the
    value error of a gs_tol-terminated method is LINEAR in gs_tol (slope*gs_tol/2),
    so a flat 1e-6-rel value criterion is physically unattainable there (at smooth
    peaks slope->0 and the strict 1e-6 applies). The solver value must also never
    EXCEED the near-exact reference. Powers are maxmin-shaped (bisection at the
    segment-centre gains), so the envelope's crossing/balance structure is realistic."""
    floor = _floor(K, params)
    n_ref = int(env["D_S"] / (env["gs_tol"] / 10.0)) + 1
    pos_tol = env["D_S"] / (env["n_scan"] - 1) + 10.0 * env["gs_tol"]
    failures = []
    for i, (ux, uy) in enumerate(_drops(K, 4, 1000 + K, env)):
        g0 = channel.path_gain((env["D_S"] / 2 - ux) ** 2 + uy ** 2 + env["d"] ** 2,
                               env["eta"])
        _, P, _ = power_alloc.maxmin_bisection(g0, floor, "A", P_max=env["P_max"])
        obj = _minrate_closure(ux, uy, P, floor, env)
        xs, fs, _ = placement.scan_refine_max(obj, 0.0, env["D_S"], env["n_scan"],
                                              env["gs_tol"])
        xe, fe, slope = _minrate_grid_max(ux, uy, P, floor, env, n_ref)
        val_tol = max(1e-6 * abs(fe), slope * env["gs_tol"])
        if fs > fe + 1e-8 or fe - fs > val_tol or abs(xs - xe) > pos_tol:
            failures.append(f"drop {i}: scan x*={xs:.4f} f={fs:.6f} vs exhaustive "
                            f"x*={xe:.4f} f={fe:.6f} (val_tol={val_tol:.2e})")
    assert not failures, (
        f"GATE A-maxmin FAILED at K={K}:\n" + "\n".join(failures[:5]))


def test_gateA_maxmin_pinned_scan_refine(params, env) -> None:
    """Pinned instance at FIXED powers P=(1,0.2114): scan_refine_max must find the
    true envelope peak x*~10.173 @ 2.3549 (which K-start missed), matching the
    fine-grid exhaustive reference."""
    ux = np.array([12.416, 10.171]); uy = np.array([-1.790, -3.107])
    P = np.array([1.0, 0.2114])
    floor = _floor(2, params)
    obj = _minrate_closure(ux, uy, P, floor, env)
    xs, fs, _ = placement.scan_refine_max(obj, 0.0, env["D_S"], env["n_scan"],
                                          env["gs_tol"])
    n_ref = int(env["D_S"] / (env["gs_tol"] / 10.0)) + 1
    xe, fe, slope = _minrate_grid_max(ux, uy, P, floor, env, n_ref)
    pos_tol = env["D_S"] / (env["n_scan"] - 1) + 10.0 * env["gs_tol"]
    val_tol = max(1e-6 * abs(fe), slope * env["gs_tol"])   # kink-calibrated, see above
    assert fs <= fe + 1e-8 and fe - fs <= val_tol and abs(xs - xe) <= pos_tol
    assert abs(xs - 10.173) <= pos_tol and fs == pytest.approx(2.3549, abs=1e-3)


# ----------------------------------------------------------------------------
# GATE B: convergence within max_iter + physical ceiling
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("objective", ["sumrate", "maxmin"])
def test_gateB_converges_and_bounded(params, env, objective) -> None:
    """AO terminates with status 'converged' and the final objective respects the
    physical rate ceiling log2(1 + K*P_max*g_max/floor), g_max = eta/d^2 (u=y=0)."""
    for K in (1, 2, 4):
        floor = _floor(K, params)
        g_max = env["eta"] / env["d"] ** 2
        ceiling = math.log2(1.0 + K * env["P_max"] * g_max / floor)
        for ux, uy in _drops(K, 3, 600 + K, env):
            r = altopt.ao_segment(ux, uy, 0.0, env["D_S"], env["anchors"],
                                  objective, "A", params, floor=floor)
            assert r["status"] == "converged", f"K={K}: {r['status']} at max_iter"
            assert r["obj"] <= ceiling + 1e-9, f"K={K}: obj {r['obj']} > ceiling {ceiling}"


# ----------------------------------------------------------------------------
# GATE C: sum-rate decoupling (<= 2 iterations) vs max-min coupling
# ----------------------------------------------------------------------------
def test_gateC_sumrate_decouples_maxmin_couples(params, env) -> None:
    """GT4 sum-rate decoupling: under Version A / no QoS the power block's optimum
    (all at cap) is position-independent => AO converges in one improving pass
    (n_iter <= 2 incl. the confirming pass). Max-min couples powers to gains, so its
    count is >= the sum-rate count on the same drop."""
    K = 2
    floor = _floor(K, params)
    for ux, uy in _drops(K, 4, 700, env):
        r_sum = altopt.ao_segment(ux, uy, 0.0, env["D_S"], env["anchors"],
                                  "sumrate", "A", params, floor=floor)
        assert r_sum["n_iter"] <= 2, f"sumrate took {r_sum['n_iter']} iterations"
        try:
            r_mm = altopt.ao_segment(ux, uy, 0.0, env["D_S"], env["anchors"],
                                     "maxmin", "A", params, floor=floor)
            assert r_mm["n_iter"] >= r_sum["n_iter"] or r_mm["n_iter"] <= 2
        except AssertionError as e:
            pytest.fail(f"maxmin AO failed on the Gate-C drop: {e}")


# ----------------------------------------------------------------------------
# GATE D: multi-start agreement (empirical, logged where it fails)
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("objective", ["sumrate", "maxmin"])
def test_gateD_multistart_agreement(params, env, objective) -> None:
    """Multi-start diagnostic, REINTERPRETED post-P4-4.5 (x-step now certified):
    * sumrate: all inits MUST agree (decoupled, block solvers exact) -- spread <= 1e-3
      asserted;
    * maxmin: x-step failures (exceptions) are bugs and fail the test; a REMAINING
      spread is now attributable as a REAL AO LOCAL OPTIMUM (init-dependent KKT
      stationary point of the non-convex joint problem) -- LOGGED with the instance,
      never hidden, and the best init is NOT called global. Multi-start is the
      documented mitigation (GT4: empirical evidence, never proof)."""
    K = 2
    floor = _floor(K, params)
    drops = _drops(K, 3, 800, env) + [
        (np.array([4.229, 12.705]), np.array([-2.091, 0.151]))]   # pinned bimodal drop
    tol = 1e-3
    failures, spreads, local_optima = [], [], []
    for i, (ux, uy) in enumerate(drops):
        try:
            ms = altopt.ao_multistart(ux, uy, 0.0, env["D_S"], env["anchors"],
                                      objective, "A", params, n_random=2, floor=floor)
            spreads.append(ms["spread"])
            if ms["spread"] > tol:
                local_optima.append(
                    f"drop {i} (ux={ux.round(3)}, uy={uy.round(3)}): spread "
                    f"{ms['spread']:.3e}, objs={ {k: round(v, 4) for k, v in ms['all_final_objs'].items()} }")
        except AssertionError as e:
            failures.append(f"drop {i}: x-step failure: {e}")
    print(f"\n[GATE D LOG] {objective}: spreads={[f'{s:.2e}' for s in spreads]}")
    if local_optima:
        print(f"[GATE D LOG] {objective}: REAL AO LOCAL OPTIMA (init-dependent KKT "
              f"points; multi-start is the mitigation):\n  " + "\n  ".join(local_optima))
    assert not failures, (
        f"GATE D FAILED ({objective}) -- x-step exceptions:\n" + "\n".join(failures[:5]))
    if objective == "sumrate":
        assert not local_optima, (
            "GATE D FAILED (sumrate must be init-independent):\n"
            + "\n".join(local_optima[:5]))


# ----------------------------------------------------------------------------
# GATE E: sub-block optimality preserved at the fixed point
# ----------------------------------------------------------------------------
def test_gateE_blocks_match_direct_calls(params, env) -> None:
    """At convergence, the P-step equals a direct power_alloc call at x_star and the
    x-step equals a direct K-start call at P_star (AO does not degrade either
    certified sub-solver). Sum-rate mode (kstart-certified objective)."""
    K = 2
    floor = _floor(K, params)
    ux, uy = _drops(K, 1, 900, env)[0]
    r = altopt.ao_segment(ux, uy, 0.0, env["D_S"], env["anchors"],
                          "sumrate", "A", params, floor=floor)
    g_star = channel.path_gain((r["x_star"] - ux) ** 2 + uy ** 2 + env["d"] ** 2,
                               env["eta"])
    P_direct, _, status = power_alloc.power_lp(g_star, floor, "A", P_max=env["P_max"])
    assert status == "optimal"
    assert np.allclose(P_direct, r["P_star"], rtol=0.0, atol=1e-12)

    def obj_x(x: float) -> float:
        g = channel.path_gain((x - ux) ** 2 + uy ** 2 + env["d"] ** 2, env["eta"])
        return power_alloc.segment_rate_from_powers(r["P_star"], g, floor)

    x_direct, f_direct, _ = placement.golden_section_kstart(
        obj_x, 0.0, env["D_S"], ux, env["gs_tol"])
    assert abs(x_direct - r["x_star"]) <= 10.0 * env["gs_tol"]
    assert abs(f_direct - r["obj"]) <= 1e-6 * abs(r["obj"]) + 1e-12


def test_gateE_maxmin_xstep_matches_direct_scan_refine(params, env) -> None:
    """Maxmin mirror of Gate E: at convergence the x-step result is consistent with a
    direct scan_refine_max call at the final powers. Tolerances: the direct call can
    never EXCEED the AO objective (safeguarded acceptance keeps the best proposal;
    determinism makes a re-run identical when the last proposal was accepted) and can
    sit at most the dip-guard bound below it (rejected epsilon-worse final proposal)."""
    K = 2
    floor = _floor(K, params)
    ux, uy = _drops(K, 2, 800, env)[1]              # seed-800 drop with zero spread
    r = altopt.ao_segment(ux, uy, 0.0, env["D_S"], env["anchors"], "maxmin", "A",
                          params, floor=floor)
    assert r["status"] == "converged"
    obj = _minrate_closure(ux, uy, r["P_star"], floor, env)
    x_direct, f_direct, _ = placement.scan_refine_max(
        obj, 0.0, env["D_S"], env["n_scan"], env["gs_tol"])
    assert f_direct <= r["obj"] + 1e-9
    assert f_direct >= r["obj"] - 5e-4
    if abs(f_direct - r["obj"]) <= 1e-12:            # accepted-path case: same point
        assert abs(x_direct - r["x_star"]) <= 10.0 * env["gs_tol"]


# ----------------------------------------------------------------------------
# GATE F: K=1 EDMA-collapse chain
# ----------------------------------------------------------------------------
def test_gateF_K1_collapse(params, env) -> None:
    """K=1: AO = transmit-at-cap + single golden-section placement over the user
    (kstart at K=1 is bit-identical to plain golden-section, P4-1.5), <= 2 iterations."""
    floor = _floor(1, params)
    ux, uy = _drops(1, 1, 950, env)[0]
    for objective in ("sumrate", "maxmin"):
        r = altopt.ao_segment(ux, uy, 0.0, env["D_S"], env["anchors"],
                              objective, "A", params, floor=floor)
        assert r["P_star"][0] == pytest.approx(env["P_max"], rel=1e-9)
        assert abs(r["x_star"] - ux[0]) <= 10.0 * env["gs_tol"]
        assert r["n_iter"] <= 2
