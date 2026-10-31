"""Phase-4 P4-1 verification gates A-F for src/placement.py. Sources: GT4 (CLAUDE.md Sec. 9).

Conventions under test (P4-1 locked): ABSOLUTE segment grid; TRUE instantaneous rate
with CONSTANT floor I_m + N0; home links pure LoS. NOTE: because the home link carries
no blockage factor, the objective is phi-INDEPENDENT -- phi enters only through the
EDMA Lemma 4 flag (unimodality_flag), which is exercised in Gate D.

The floor is built as N0 + 2*K*P*L1_exact (the Phase-3 mean interference E[I_m], the
physically-motivated constant); the OPTIMIZER's argmax is floor-independent, the floor
only scales the rate.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src import analytic, placement
from src.config import load_params

# Deterministic drop counts (OURS, test-only knobs).
_N_DROPS_GATE_A = 20
_N_DROPS_GATE_D = 12
_DENSE_GRID = 8001


@pytest.fixture(scope="module")
def params() -> dict:
    return load_params()


@pytest.fixture(scope="module")
def phys(params) -> dict:
    """Physical scalars + Phase-4 knobs shared by all gates."""
    p3 = params["phase3"]
    p4 = params["phase4"]
    D_S = float(params["D_S"])
    tol = float(p4["gs_tol"])
    # Exhaustive grid: >= exhaustive_points, or step gs_tol/10 if finer (params.yaml).
    n_ex = max(int(p4["exhaustive_points"]), int(math.ceil(D_S / (tol / 10.0))) + 1)
    return {
        "eta": float(params["eta"]), "phi": float(params["phi"]), "d": float(params["d"]),
        "D_S": D_S, "D_W": float(params["D_W"]), "D_L": float(params["D_L"]),
        "P": float(p3["P"]), "N0": float(p3["N0"]),
        "tol": tol, "n_ex": n_ex, "L1": analytic.L1_exact(params),
    }


def _floor(K: int, ph: dict) -> float:
    """Constant floor I_m + N0 with I_m = E[I_m] = 2*K*P*L1_exact (GT4 true rate)."""
    return ph["N0"] + 2.0 * K * ph["P"] * ph["L1"]


def _drop(rng: np.random.Generator, K: int, ph: dict) -> tuple[np.ndarray, np.ndarray]:
    """Uniform K-user drop over segment [0, D_S] x [-D_W/2, D_W/2] (SM user model)."""
    ux = rng.uniform(0.0, ph["D_S"], K)
    uy = rng.uniform(-ph["D_W"] / 2.0, ph["D_W"] / 2.0, K)
    return ux, uy


def _objective(ux: np.ndarray, uy: np.ndarray, ph: dict, floor: float):
    return lambda x: placement.rate_at_position(x, ux, uy, ph["eta"], ph["d"], ph["P"], floor)


def _count_humps(f: np.ndarray) -> int:
    """Number of interior maxima: +->- sign changes of the discrete derivative."""
    df = np.diff(f)
    s = np.sign(df[np.abs(df) > 1e-13])
    return int(np.sum((s[:-1] > 0) & (s[1:] < 0)))


# ----------------------------------------------------------------------------
# Sanity: repo convention (squared distance into channel.path_gain)
# ----------------------------------------------------------------------------
def test_collected_power_matches_direct_formula(phys) -> None:
    """G_m(x) = sum_k P*eta/((x-x_k)^2 + y_k^2 + d^2)  (GT4 channel gain, EDMA Eq. 2)."""
    ph = phys
    ux = np.array([2.0, 10.0]); uy = np.array([-1.5, 3.0]); x = 5.0
    direct = sum(ph["P"] * ph["eta"] / ((x - xk) ** 2 + yk ** 2 + ph["d"] ** 2)
                 for xk, yk in zip(ux, uy))
    got = placement.collected_power(x, ux, uy, ph["eta"], ph["d"], ph["P"])
    assert got == pytest.approx(direct, rel=1e-12)
    # vectorized x agrees with scalar loop
    grid = np.array([0.5, 5.0, 12.0])
    vec = placement.collected_power(grid, ux, uy, ph["eta"], ph["d"], ph["P"])
    for i, xi in enumerate(grid):
        assert vec[i] == pytest.approx(
            placement.collected_power(float(xi), ux, uy, ph["eta"], ph["d"], ph["P"]), rel=1e-12)


# ----------------------------------------------------------------------------
# GATE A (HEADLINE): golden-section <-> exhaustive across drops, K in {1,2,4}
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("K", [1, 2, 4])
def test_gateA_kstart_matches_exhaustive(phys, K: int) -> None:
    """K-start optimum == exhaustive global optimum (pos <= 10*tol, rate rel 1e-6).

    Certifies the P4-1.5 basin-decomposition solver reaches the TRUE GLOBAL optimum of
    the TRUE rate (GT4 K-start algorithm), including on multimodal K>=2 drops where
    plain golden-section failed in P4-1.
    """
    ph = phys
    rng = np.random.default_rng(1000 + K)
    floor = _floor(K, ph)
    failures = []
    for t in range(_N_DROPS_GATE_A):
        ux, uy = _drop(rng, K, ph)
        obj = _objective(ux, uy, ph, floor)
        xg, fg, _ = placement.golden_section_kstart(obj, 0.0, ph["D_S"], ux, ph["tol"])
        xe, fe = placement.exhaustive_max(obj, 0.0, ph["D_S"], ph["n_ex"])
        if abs(xg - xe) > 10.0 * ph["tol"] or abs(fg - fe) > 1e-6 * abs(fe):
            failures.append(
                f"drop {t}: users_x={ux.round(3)} users_y={uy.round(3)} | "
                f"kstart x*={xg:.4f} rate={fg:.6f} vs exhaustive x*={xe:.4f} rate={fe:.6f} "
                f"(deficit {100 * (fe - fg) / fe:.2f}%)")
    assert not failures, (
        f"GATE A FAILED at K={K}: K-start missed the global optimum on "
        f"{len(failures)}/{_N_DROPS_GATE_A} drops:\n" + "\n".join(failures[:5]))


def test_gateA_known_multimodal_drop_K2(phys) -> None:
    """Pinned P4-1 counterexample (K=2, users x={4.229,12.705}): the TRUE rate is
    BIMODAL and plain golden-section converged to the wrong hump (x*=4.48 @ 5.034 bits
    vs global x*=12.61 @ 5.510 bits). golden_section_kstart must now find the global
    optimum (GT4 K-start algorithm) -- kept as a named regression."""
    ph = phys
    ux = np.array([4.229, 12.705]); uy = np.array([-2.091, 0.151])
    floor = _floor(2, ph)
    obj = _objective(ux, uy, ph, floor)
    xg, fg, _ = placement.golden_section_kstart(obj, 0.0, ph["D_S"], ux, ph["tol"])
    xe, fe = placement.exhaustive_max(obj, 0.0, ph["D_S"], ph["n_ex"])
    assert abs(xg - xe) <= 10.0 * ph["tol"] and abs(fg - fe) <= 1e-6 * abs(fe), (
        f"GATE A FAILED (known K=2 bimodal drop): kstart x*={xg:.4f} rate={fg:.6f} vs "
        f"exhaustive x*={xe:.4f} rate={fe:.6f} -- rate deficit {100 * (fe - fg) / fe:.2f}%")
    # and it is the P4-1-recorded global hump, not the wrong one plain GS chose
    assert abs(xg - 12.61) < 0.05, f"expected the x~12.61 hump, got {xg:.4f}"
    assert fg == pytest.approx(5.5095, abs=1e-3)


# ----------------------------------------------------------------------------
# GATE B: closed form <-> solver (symmetric regime), and clamping
# ----------------------------------------------------------------------------
def test_gateB_symmetric_midpoint_matches_gs(phys) -> None:
    """Symmetric y=0 drop (users at centre +- 1 m, unimodal): golden-section optimum ==
    midpoint_placement (EDMA Remark 8) within tol.
    """
    ph = phys
    c = ph["D_S"] / 2.0
    ux = np.array([c - 1.0, c + 1.0]); uy = np.array([0.0, 0.0])
    # anchors = neighbouring segment centres; delta^2*phi = D_S^2*phi >= 1 holds (flag).
    x_left, x_right = c - ph["D_S"], c + ph["D_S"]
    assert placement.unimodality_flag(
        placement.delta_from_anchors(x_left, x_right), ph["phi"])
    mid = placement.midpoint_placement(x_left, x_right, 0.0, ph["D_S"])
    assert mid == pytest.approx(c)
    obj = _objective(ux, uy, ph, _floor(2, ph))
    xg, _, _ = placement.golden_section(obj, 0.0, ph["D_S"], ph["tol"])
    assert abs(xg - mid) <= ph["tol"], f"gs {xg} vs midpoint {mid}"


def test_gateB_midpoint_clamps_to_boundary(phys) -> None:
    """Midpoint outside [x_lo, x_hi] => nearer boundary (EDMA Remark 8 boundary case)."""
    D_S = phys["D_S"]
    assert placement.midpoint_placement(D_S + 1.0, 3.0 * D_S, 0.0, D_S) == D_S
    assert placement.midpoint_placement(-3.0 * D_S, -1.0, 0.0, D_S) == 0.0


# ----------------------------------------------------------------------------
# GATE C: K=1 -> EDMA collapse (mandatory cross-consistency)
# ----------------------------------------------------------------------------
def test_gateC_K1_optimum_over_user(phys) -> None:
    """K=1: single Lorentzian bump; the interior optimum is the over-the-user position
    x* = x_u (EDMA single-user placement, Algorithm-1 case). Golden-section must land on
    it (10*tol).
    """
    ph = phys
    rng = np.random.default_rng(7)
    floor = _floor(1, ph)
    for _ in range(8):
        ux, uy = _drop(rng, 1, ph)
        obj = _objective(ux, uy, ph, floor)
        xg, _, _ = placement.golden_section(obj, 0.0, ph["D_S"], ph["tol"])
        assert abs(xg - ux[0]) <= 10.0 * ph["tol"], f"gs {xg} vs user {ux[0]}"


def test_gateC_K1_kstart_equals_plain_golden_section(phys) -> None:
    """EDMA collapse of the ALGORITHM: at K=1 golden_section_kstart builds exactly one
    bracket [x_lo, x_hi], so its result is BIT-IDENTICAL to a single golden_section
    call (same probes, same floats) -- the K-general algorithm degenerates to EDMA
    Algorithm 1 (GT4 K-start item)."""
    ph = phys
    rng = np.random.default_rng(17)
    floor = _floor(1, ph)
    for _ in range(5):
        ux, uy = _drop(rng, 1, ph)
        obj = _objective(ux, uy, ph, floor)
        xk, fk, nk = placement.golden_section_kstart(obj, 0.0, ph["D_S"], ux, ph["tol"])
        xg, fg, ng = placement.golden_section(obj, 0.0, ph["D_S"], ph["tol"])
        assert xk == xg and fk == fg and nk == ng


def test_gateC_K1_objective_is_edma_single_user_form(phys) -> None:
    """Structural reduction: at K=1 the TRUE rate equals the EDMA single-user rate
    log2(1 + P*eta/((x-x_u)^2 + y_u^2 + d^2)/floor) exactly (no extra terms)."""
    ph = phys
    ux = np.array([3.7]); uy = np.array([-2.2])
    floor = _floor(1, ph)
    for x in (0.0, 2.5, 7.9, ph["D_S"]):
        expected = math.log2(
            1.0 + ph["P"] * ph["eta"] / ((x - ux[0]) ** 2 + uy[0] ** 2 + ph["d"] ** 2) / floor)
        got = placement.rate_at_position(x, ux, uy, ph["eta"], ph["d"], ph["P"], floor)
        assert got == pytest.approx(expected, rel=1e-12)


# ----------------------------------------------------------------------------
# GATE D: unimodality sanity (assert where flag True; LOG where flag False)
# ----------------------------------------------------------------------------
def test_gateD_K1_single_hump_where_flag_true(phys) -> None:
    """EDMA Lemma 4 is K=1-ONLY (P4-1 finding, GT4): where delta^2*phi >= 1, assert the
    single hump for K=1 ONLY -- the single-Lorentzian case Lemma 4 actually covers."""
    ph = phys
    # anchors = neighbouring segment centres => delta = D_S; flag True at phi=0.02.
    delta = placement.delta_from_anchors(-ph["D_S"] / 2.0, 3.0 * ph["D_S"] / 2.0)
    assert placement.unimodality_flag(delta, ph["phi"])
    rng = np.random.default_rng(2001)
    floor = _floor(1, ph)
    grid = np.linspace(0.0, ph["D_S"], _DENSE_GRID)
    for t in range(_N_DROPS_GATE_D):
        ux, uy = _drop(rng, 1, ph)
        f = placement.rate_at_position(grid, ux, uy, ph["eta"], ph["d"], ph["P"], floor)
        n = _count_humps(f)
        assert n == 1, f"K=1 drop {t}: {n} humps (users_x={ux.round(3)})"


@pytest.mark.parametrize("K", [2, 4])
def test_gateD_K_ge_2_global_optimality_on_multimodal_drops(phys, K: int) -> None:
    """K>=2 re-scope (P4-1.5): unimodality is NOT asserted (it is FALSE -- the P4-1
    finding; phi never enters the pure-LoS objective). The certified property is now
    GLOBAL OPTIMALITY: golden_section_kstart must match exhaustive_max on every drop,
    INCLUDING the multimodal ones (whose presence in the sample is asserted so this
    test provably exercises the failure mode plain golden-section had)."""
    ph = phys
    rng = np.random.default_rng(2000 + K)   # same seeds as the P4-1 red run
    floor = _floor(K, ph)
    grid = np.linspace(0.0, ph["D_S"], _DENSE_GRID)
    n_multimodal = 0
    failures = []
    for t in range(_N_DROPS_GATE_D):
        ux, uy = _drop(rng, K, ph)
        f = placement.rate_at_position(grid, ux, uy, ph["eta"], ph["d"], ph["P"], floor)
        if _count_humps(f) > 1:
            n_multimodal += 1
        obj = _objective(ux, uy, ph, floor)
        xg, fg, _ = placement.golden_section_kstart(obj, 0.0, ph["D_S"], ux, ph["tol"])
        xe, fe = placement.exhaustive_max(obj, 0.0, ph["D_S"], ph["n_ex"])
        if abs(xg - xe) > 10.0 * ph["tol"] or abs(fg - fe) > 1e-6 * abs(fe):
            failures.append(f"drop {t}: kstart x*={xg:.4f} vs exhaustive x*={xe:.4f}")
    assert n_multimodal >= 1, "sample contained no multimodal drop; test exercises nothing"
    assert not failures, (
        f"GATE D FAILED at K={K}: K-start missed the global optimum on "
        f"{len(failures)}/{_N_DROPS_GATE_D} drops ({n_multimodal} multimodal):\n"
        + "\n".join(failures[:5]))


def test_gateD_log_modality_where_flag_false(phys) -> None:
    """Where delta^2*phi < 1 the flag makes NO claim: LOG the empirical modality (do not
    assert). phi enters only the flag -- the objective itself is phi-free (pure-LoS home)."""
    ph = phys
    phi_small = 0.005
    delta = placement.delta_from_anchors(-ph["D_S"] / 2.0, 3.0 * ph["D_S"] / 2.0)
    assert not placement.unimodality_flag(delta, phi_small)
    rng = np.random.default_rng(3000)
    grid = np.linspace(0.0, ph["D_S"], _DENSE_GRID)
    counts: dict[int, int] = {}
    for K in (1, 2, 4):
        floor = _floor(K, ph)
        uni = 0
        for _ in range(_N_DROPS_GATE_D):
            ux, uy = _drop(rng, K, ph)
            f = placement.rate_at_position(grid, ux, uy, ph["eta"], ph["d"], ph["P"], floor)
            uni += int(_count_humps(f) == 1)
        counts[K] = uni
    print(f"\n[GATE D LOG] flag False (phi={phi_small}): empirically unimodal drops "
          f"per K over {_N_DROPS_GATE_D}: {counts} (recorded, not asserted)")


# ----------------------------------------------------------------------------
# GATE E: separability (uplink independence)
# ----------------------------------------------------------------------------
def test_gateE_separability_neighbour_perturbation(phys) -> None:
    """Perturbing OTHER segments' users (hence their antennas) leaves segment m's
    optimum bit-identical: the M-segment problem is M independent 1-D problems (GT4)."""
    ph = phys
    M, K = 3, 2
    D_S = ph["D_L"] / M
    rng = np.random.default_rng(11)
    ux = np.stack([rng.uniform(m * D_S, (m + 1) * D_S, K) for m in range(M)])
    uy = rng.uniform(-ph["D_W"] / 2.0, ph["D_W"] / 2.0, (M, K))
    floor = _floor(K, ph)
    base = placement.optimize_all_segments(
        ux, uy, ph["eta"], ph["d"], ph["P"], floor, ph["D_L"], M, ph["tol"])

    ux2, uy2 = ux.copy(), uy.copy()
    for j in (0, 2):                                  # perturb both neighbours of m=1
        ux2[j] = rng.uniform(j * D_S, (j + 1) * D_S, K)
        uy2[j] = rng.uniform(-ph["D_W"] / 2.0, ph["D_W"] / 2.0, K)
    pert = placement.optimize_all_segments(
        ux2, uy2, ph["eta"], ph["d"], ph["P"], floor, ph["D_L"], M, ph["tol"])

    assert pert[1]["x_star"] == base[1]["x_star"]     # bit-identical, not approx
    assert pert[1]["f_star"] == base[1]["f_star"]
    assert pert[1]["n_iter"] == base[1]["n_iter"]
    # and the perturbed segments DID change (the perturbation was real)
    assert pert[0]["x_star"] != base[0]["x_star"] or pert[2]["x_star"] != base[2]["x_star"]


# ----------------------------------------------------------------------------
# GATE F: probe reuse + O(log 1/eps) iteration count
# ----------------------------------------------------------------------------
def test_gateF_one_new_eval_per_iteration(phys) -> None:
    """EDMA Algorithm 1 efficiency: exactly ONE new objective evaluation per iteration
    after the 2-eval initialization (+1 final midpoint eval) => total = n_iter + 3."""
    ph = phys
    ux = np.array([5.0]); uy = np.array([1.0])
    floor = _floor(1, ph)
    calls = 0
    inner = _objective(ux, uy, ph, floor)

    def counting(x: float) -> float:
        nonlocal calls
        calls += 1
        return inner(x)

    _, _, n_iter = placement.golden_section(counting, 0.0, ph["D_S"], ph["tol"])
    assert calls == n_iter + 3, f"{calls} evals for {n_iter} iters (expected n_iter+3)"


def test_gateF_iteration_count_is_log_eps(phys) -> None:
    """n_iter <= ceil(log(width/tol)/log(1/tau)) + 1: the O(log 1/eps) claim, and
    M segments cost M x that (separability => optimize_all_segments is M gs calls)."""
    ph = phys
    tau = (math.sqrt(5.0) - 1.0) / 2.0
    ux = np.array([5.0]); uy = np.array([1.0])
    obj = _objective(ux, uy, ph, _floor(1, ph))
    _, _, n_iter = placement.golden_section(obj, 0.0, ph["D_S"], ph["tol"])
    bound = math.ceil(math.log(ph["D_S"] / ph["tol"]) / math.log(1.0 / tau)) + 1
    assert n_iter <= bound, f"n_iter={n_iter} exceeds O(log 1/eps) bound {bound}"


@pytest.mark.parametrize("K", [1, 2, 4])
def test_gateF_kstart_eval_count_and_MK_log_bound(phys, K: int) -> None:
    """K-start efficiency (GT4 K-start algorithm): per BRACKET the golden-section worker
    makes one new eval per iteration + 3 (2 init + 1 final midpoint), so with <= K
    brackets total evals == total_iters + 3*n_brackets, and total_iters <= K * the
    single-bracket O(log 1/eps) bound => O(M K log(1/eps)) over M segments."""
    ph = phys
    tau = (math.sqrt(5.0) - 1.0) / 2.0
    rng = np.random.default_rng(4000 + K)
    ux, uy = _drop(rng, K, ph)
    floor = _floor(K, ph)
    n_brackets = len(np.unique(ux))         # midpoint splits: one bracket per distinct user
    calls = 0
    inner = _objective(ux, uy, ph, floor)

    def counting(x: float) -> float:
        nonlocal calls
        calls += 1
        return inner(x)

    _, _, total_iters = placement.golden_section_kstart(
        counting, 0.0, ph["D_S"], ux, ph["tol"])
    assert calls == total_iters + 3 * n_brackets, (
        f"{calls} evals for {total_iters} iters over {n_brackets} brackets")
    per_bracket_bound = math.ceil(math.log(ph["D_S"] / ph["tol"]) / math.log(1.0 / tau)) + 1
    assert total_iters <= K * per_bracket_bound, (
        f"total_iters={total_iters} exceeds K x O(log 1/eps) = {K * per_bracket_bound}")
