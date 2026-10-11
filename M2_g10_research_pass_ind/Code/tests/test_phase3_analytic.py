"""Phase-3 analytic tests (P3-1). Each pins one invariant / equation from CLAUDE.md
Sec. 7. The Exit-Gate-3 Monte-Carlo comparison stays skipped until P3-2.
"""
from __future__ import annotations

import copy
import math

import pytest
from scipy.integrate import quad

import numpy as np

from src import analytic as A, montecarlo as MC
from src.config import load_params


@pytest.fixture(scope="module")
def params() -> dict:
    return load_params()


def _at_rho_db(p: dict, rho_db: float) -> dict:
    """Copy of params with N0 = P/rho (fixed P) -> the load-time rho==P/N0 stays valid."""
    q = copy.deepcopy(p)
    q["phase3"]["N0"] = float(q["phase3"]["P"]) / (10.0 ** (rho_db / 10.0))
    return q


# R_tilde (Jensen mean-interference rate) was lifted to analytic.R_tilde in P3-4
# (one source of truth, shared with the windfall-decomposition driver); tests call it.


def _phys(p):
    p3 = p["phase3"]
    return (float(p["eta"]), float(p["phi"]), float(p["d"]), float(p["D_S"]),
            float(p["D_W"]), float(p["delta"]), float(p3["P"]), float(p3["N0"]))


# --- transforms g(z), h(z) ---
def test_g_at_zero_is_one(params) -> None:
    assert A.g_transform(0.0, params) == pytest.approx(1.0, abs=1e-12)
    assert A.h_transform(0.0, params) == pytest.approx(1.0, abs=1e-12)


def test_g_h_bounds_0_to_1(params) -> None:
    # moderate z grid (below the g-underflow scale), strict 0 < g,h <= 1.
    for z in (1e3, 1e5, 1e6, 1e7, 5e7):
        g, h = A.g_transform(z, params), A.h_transform(z, params)
        assert 0.0 < g <= 1.0
        assert 0.0 < h <= 1.0


# --- removable singularity ---
def test_removable_singularity_limit_equals_KPGbar(params) -> None:
    eta, phi, d, D_S, D_W, delta, P, N0 = _phys(params)
    Gbar = A.G_bar(params)
    z = 1e-6 * d * d / (P * eta)            # small-z (linear regime), > 0
    for K in (1, 5, 20):
        g, h = A.g_transform(z, params), A.h_transform(z, params)
        F = (1.0 / z) * math.exp(-z * N0) * (h ** (2 * K)) * (-math.expm1(K * math.log(g)))
        assert F == pytest.approx(K * P * Gbar, rel=1e-4)


def test_ergodic_rate_returns_truncation_error(params) -> None:
    tol = float(params["phase3"]["z_quad_tol"])
    for K in (1, 5, 50):
        rate, te = A.ergodic_rate_exact(K, params)
        assert math.isfinite(rate) and math.isfinite(te)
        assert 0.0 <= te < tol


def test_ergodic_rate_nonnegative(params) -> None:
    for K in (1, 5, 20, 50):
        rate, _ = A.ergodic_rate_exact(K, params)
        assert rate >= 0.0


# --- derivative identities (central FD, eps = s*P*eta/d^2 = 1e-3) ---
def test_gprime0_equals_P_Gbar(params) -> None:
    eta, phi, d, D_S, D_W, delta, P, N0 = _phys(params)
    s = 1e-3 * d * d / (P * eta)
    gp = (A.g_transform(s, params) - A.g_transform(-s, params)) / (2.0 * s)  # g'(0)
    assert -gp == pytest.approx(P * A.G_bar(params), rel=1e-3)


def test_hprime0_equals_P_L1exact(params) -> None:
    eta, phi, d, D_S, D_W, delta, P, N0 = _phys(params)
    s = 1e-3 * d * d / (P * eta)
    hp = (A.h_transform(s, params) - A.h_transform(-s, params)) / (2.0 * s)  # h'(0)
    assert -hp == pytest.approx(P * A.L1_exact(params), rel=1e-3)


# --- mean constants ---
def test_l1_exact_le_lbar_phase2(params) -> None:
    assert A.L1_exact(params) <= A.L_bar(1, params, float(params["delta"]))


def test_Gbar_greater_than_Gbar_full(params) -> None:
    assert A.G_bar(params) > A.G_bar_full(params)


def test_Gbar_inf_closed_form_and_limits(params) -> None:
    eta, phi, d, D_S, D_W, delta, P, N0 = _phys(params)
    assert A.G_bar_inf(params) == pytest.approx(
        (2.0 * eta) / (d * D_W) * math.atan(D_W / (2.0 * d)), rel=1e-12)
    small = copy.deepcopy(params); small["D_W"] = 1e-4
    assert A.G_bar_inf(small) == pytest.approx(eta / d**2, rel=1e-6)   # D_W -> 0
    big = copy.deepcopy(params); big["D_W"] = 1e8
    assert A.G_bar_inf(big) < 1e-6 * eta / d**2                        # D_W -> inf -> 0


# --- Jensen backstop R_tilde <= R_bar ---
def test_jensen_Rtilde_le_Rbar(params) -> None:
    eta, phi, d, D_S, D_W, delta, P, N0 = _phys(params)
    Gbar, L1 = A.G_bar(params), A.L1_exact(params)
    ln2 = math.log(2.0)
    for K in (1, 5):
        c = N0 + 2.0 * K * P * L1                    # deterministic interference+noise
        z0 = 1e-6 * d * d / (P * eta)
        def integ(w, K=K, c=c, z0=z0):
            z = math.exp(w)
            g = A.g_transform(z, params)
            omg = 1.0 if g <= 0.0 else -math.expm1(K * math.log(g))
            return (1.0 / z) * math.exp(-z * c) * omg * z   # *z for the log-substitution
        main, _ = quad(integ, math.log(z0), math.log(50.0 / c), limit=256)
        Rtilde = (K * P * Gbar * z0 + main) / ln2
        Rbar, _ = A.ergodic_rate_exact(K, params)
        assert Rtilde <= Rbar + 1e-6


# --- gain factor ---
def test_gain_factor_in_1_to_K(params) -> None:
    for rho_db in (60.0, 80.0, 100.0, 120.0):           # rho-grid via N0 = P/rho
        q = copy.deepcopy(params)
        q["phase3"]["N0"] = float(q["phase3"]["P"]) / (10.0 ** (rho_db / 10.0))
        for K in params["K_list"]:
            g = A.gain_factor(K, q)
            assert 1.0 - 1e-9 <= g <= K + 1e-9


def test_gain_factor_limits(params) -> None:
    K = 20
    L1 = A.L1_exact(params)
    noise = copy.deepcopy(params); noise["phase3"]["N0"] = 1e12 * 2.0 * L1   # N0 >> leakage
    assert A.gain_factor(K, noise) == pytest.approx(float(K), rel=1e-3)      # -> K
    interf = copy.deepcopy(params); interf["phase3"]["N0"] = 1e-12 * 2.0 * L1  # N0 << leakage
    assert A.gain_factor(K, interf) == pytest.approx(1.0, rel=1e-3)          # -> 1


# --- system rates / power model ---
def test_system_rates_requires_power_model(params) -> None:
    with pytest.raises(ValueError):
        A.system_rates("X", 5, 3.0, params)
    with pytest.raises(ValueError):
        A.system_rates(None, 5, 3.0, params)  # type: ignore[arg-type]


# --- REGRESSION vs EDMA's PUBLISHED Table I (the test that would have caught both prior
#     TDMA errors: assert MATCH-TO-PUBLISHED-NUMBERS, not formula properties) ---
_EDMA_TABLE_I = {  # 30 dBm, phi=0.02, D_L=4*D_W; M -> ergodic uplink sum rate
    "PTDMA": {3: 5.69, 5: 3.56, 7: 2.61, 9: 2.07, 11: 1.72},
    "CTDMA": {3: 5.41, 5: 3.48, 7: 2.57, 9: 2.05, 11: 1.72},
    "EDMA":  {3: 41.07, 5: 51.80},   # NOMA row, validated within the gate (M<=M_gate~3-5)
}


def _edma_params(params):
    """EDMA Fig.3/Table I params: rho=1e12 (30 dBm/-90 dBm), D_L=4*D_W."""
    q = copy.deepcopy(params)
    q["phase3"]["N0"] = float(q["phase3"]["P"]) / 1e12
    q["D_L"] = 4.0 * float(q["D_W"])
    return q


def test_ptdma_matches_edma_table_I(params) -> None:
    q = _edma_params(params)
    for M, ref in _EDMA_TABLE_I["PTDMA"].items():
        ours = A.tdma_rate_eq9("B", 1, float(M), q, placement="u0")  # K=1, boost M, u=0
        assert ours == pytest.approx(ref, abs=0.02), f"P-TDMA M={M}: {ours:.3f} vs {ref}"


def test_ctdma_matches_edma_table_I(params) -> None:
    q = _edma_params(params)
    for M, ref in _EDMA_TABLE_I["CTDMA"].items():
        ours = A.tdma_rate_eq9("B", 1, float(M), q, placement="uavg")  # seg-centre antenna
        assert ours == pytest.approx(ref, abs=0.02), f"C-TDMA M={M}: {ours:.3f} vs {ref}"


def test_noma_row_matches_edma_table_I(params) -> None:
    # NOMA/EDMA row at K=1, u=0, edge-aware, within the adjacency gate (M=3,5). MC, fixed seed.
    q = _edma_params(params)
    eta, phi, d, D_S, D_W, delta, P, N0 = _phys(q)
    D_L = float(q["D_L"])
    for M, ref in _EDMA_TABLE_I["EDMA"].items():
        rng = np.random.default_rng(1)
        N = 150000
        D_Sm = D_L / M
        x = np.stack([rng.uniform(m * D_Sm, (m + 1) * D_Sm, N) for m in range(M)])
        y = rng.uniform(-D_W / 2, D_W / 2, (M, N))
        tot = np.zeros(N)
        for m in range(M):
            S = P * eta / (y[m] ** 2 + d**2)              # u=0 (PA at served user's x)
            I = np.zeros(N)
            for j in (m - 1, m + 1):
                if 0 <= j < M:
                    R2 = (x[j] - x[m]) ** 2 + y[j] ** 2 + d**2
                    coin = rng.random(N) < np.exp(-phi * R2)
                    I += np.where(coin, P * eta / R2, 0.0)
            tot += np.log2(1 + S / (N0 + I))
        ours = float(tot.mean())
        assert ours == pytest.approx(ref, rel=0.02), f"EDMA(NOMA) M={M}: {ours:.2f} vs {ref}"


def test_noma_dominates_tdma_margin_grows_with_K(params) -> None:
    # Validated thesis: NOMA > TDMA at all rho; margin grows with K (TDMA time-share crushed).
    M = 3.0
    for db in (60.0, 80.0, 100.0, 120.0):
        q = _at_rho_db(params, db)
        margins = []
        for K in (1, 10, 50):
            r = A.system_rates("B", K, M, q)
            assert r["NOMA"] > r["TDMA"], f"rho={db} K={K}: NOMA {r['NOMA']:.2f} <= TDMA {r['TDMA']:.2f}"
            margins.append(r["NOMA"] / r["TDMA"])
        assert margins[2] > margins[0]   # ratio grows from K=1 to K=50


# --- Option (b): edge-aware far interference reproduces EDMA Table I across M ---
# MC samplers now live in src/montecarlo.py (one source of truth -- the Fig.4 driver
# and these regression tests call the SAME functions). Pair A = triangular (u=0);
# Pair B = fixed-delta (u-avg). Lifted verbatim in P3-4.
def test_edma_full_interference_matches_table_I(params) -> None:
    # GATE: edge-aware FULL interference reproduces EDMA Table I within ~2% at ALL 5 M.
    q = _edma_params(params)
    table = {3: 41.07, 5: 51.80, 7: 50.28, 9: 43.91, 11: 37.15}
    for M, ref in table.items():
        ours = MC.system_sum_rate_mc_triangular(M, q)["mean_rate"]
        assert ours == pytest.approx(ref, rel=0.02), f"EDMA full M={M}: {ours:.2f} vs {ref}"


def test_full_interference_analytic_matches_mc_fixed_delta(params) -> None:
    # PAIR B (Exit Gate 3 for the full-interference analytic): analytic-fixed-delta vs
    # MC-fixed-delta -- SAME physical model (u-avg home, uniform separation). Must be tight.
    q = _edma_params(params)
    for M in (3, 5, 7):
        qm = copy.deepcopy(q); qm["M"] = M; qm["D_S"] = qm["D_L"] / M; qm["delta"] = qm["D_S"] / 2
        a = A.system_sum_rate_exact(1, M, qm, home_placement="uavg")[0]
        mc = MC.system_sum_rate_mc_fixed_delta(M, qm)["mean_rate"]
        assert a == pytest.approx(mc, rel=0.01), f"Pair B M={M}: analytic {a:.2f} vs MC {mc:.2f}"


def test_optimum_in_M5_M6_plateau(params) -> None:
    # OUR fixed-delta system is non-monotone with a BROAD optimum on the M=5-6 plateau
    # (verified argmax is M=6 by ~0.4%; M=5 is a hair behind). We assert only that the
    # plateau dominates its M=3 and M=7 shoulders -- NOT that M=5 is the global argmax
    # (the triangular/EDMA model peaks at M=5; ours at M=6). Source: P3-4 data check.
    q = _edma_params(params)
    def rate(M):
        qm = copy.deepcopy(q); qm["M"] = M; qm["D_S"] = qm["D_L"] / M; qm["delta"] = qm["D_S"] / 2
        return A.system_sum_rate_exact(1, M, qm, home_placement="uavg")[0]
    r3, r5, r7 = rate(3), rate(5), rate(7)
    assert r5 > r3 and r5 > r7   # M=5-6 plateau dominates the M=3 / M=7 shoulders


def test_system_sum_rate_M3_far_negligible(params) -> None:
    # M=3 reduction premise: the far (ell>=2) product is ~1, so (b) reduces to 2-neighbour.
    q = _edma_params(params); q["M"] = 3; q["D_S"] = q["D_L"] / 3; q["delta"] = q["D_S"] / 2
    for z in (1e6, 1e10, 1e12):
        assert A.h_ell(z, 2, q) == pytest.approx(1.0, abs=1e-3)
    rate, te = A.system_sum_rate_exact(1, 3, q, home_placement="uavg")
    assert te < float(q["phase3"]["z_quad_tol"]) and rate > 0


# --- R_det reduces to EDMA single-user form at K=1 ---
def test_Rdet_reduces_to_edma_single_user(params) -> None:
    eta, phi, d, D_S, D_W, delta, P, N0 = _phys(params)
    rho = P / N0
    Gbar, L1 = A.G_bar(params), A.L1_exact(params)
    edma = math.log2(1.0 + Gbar / (1.0 / rho + 2.0 * L1))     # EDMA single-user (per-P norm)
    assert A.R_det(1, params) == pytest.approx(edma, rel=1e-12)


# --- Phase-2 boundaries reproduced ---
def test_kstar_reproduces_phase2(params) -> None:
    eta, phi, d, D_S, D_W, delta, P, N0 = _phys(params)
    rho = P / N0
    L1 = A.L1_exact(params)
    kstar = N0 / (2.0 * P * L1)
    assert kstar == pytest.approx(1.0 / (2.0 * rho * L1), rel=1e-12)   # N0/(2P L1)=1/(2 rho L1)
    # exact K* exceeds the Phase-2 bounded K_star (L1_exact < L_bar => larger K*).
    assert kstar > A.K_star(params)


def test_Mgate_formula(params) -> None:
    phi, D_L = float(params["phi"]), float(params["D_L"])
    M_gate = D_L * math.sqrt(phi / 3.0)
    assert phi * (D_L / M_gate) ** 2 == pytest.approx(3.0, rel=1e-12)   # gate: phi*D_S^2=3
    assert M_gate == pytest.approx(3.266, abs=1e-3)


# --- Exit Gate 3: ergodic_rate_exact matches ergodic_rate_mc within c*stderr ---
def test_ergodic_rate_matches_mc(params) -> None:
    c = 3.0  # agreement tolerance in standard-error units
    q = _at_rho_db(params, 80.0)            # knee-resolving rho (K* ~ 8.9, in K_mc range)
    seed = int(params["seed"]); N = int(params["N_trials"])
    for K in params["phase3"]["K_mc"]:
        Rbar, _ = A.ergodic_rate_exact(K, q)
        res = MC.ergodic_rate_mc(K, q, "A", np.random.default_rng(seed), n_trials=N)
        gap = abs(Rbar - res["mean_rate"])
        assert gap <= c * res["stderr"], (
            f"K={K}: |analytic-MC|={gap:.2e} > {c}*stderr={c*res['stderr']:.2e}")


def test_jensen_Rtilde_le_Rbar_mc(params) -> None:
    # numerical Jensen backstop: R_tilde (mean-interference) <= R_bar (true MC).
    q = _at_rho_db(params, 80.0)
    seed = int(params["seed"]); N = int(params["N_trials"])
    for K in params["phase3"]["K_mc"]:
        res = MC.ergodic_rate_mc(K, q, "A", np.random.default_rng(seed), n_trials=N)
        assert A.R_tilde(K, q) <= res["mean_rate"] + 3.0 * res["stderr"]


def test_ergodic_rate_mc_requires_power_model(params) -> None:
    with pytest.raises(ValueError):
        MC.ergodic_rate_mc(5, params, "X", np.random.default_rng(0), n_trials=10)
