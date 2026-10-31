"""Monte-Carlo validation of the far/adjacent leakage ratio.

The MC (exact 1/r^2) is the GROUND TRUTH. The analytic ``ratio_R`` froze 1/r^2 at
its per-cell floor 1/(u_min^2+d^2); that freeze makes ``ratio_R`` a per-term-bound
PROXY, NOT a rigorous upper bound on the true ratio (it runs ~1.5x below true,
because the freeze inflates the adjacent term more than the far term). See CLAUDE.md
Sec. 3. Therefore:
  * the rigorous one-directional bound is the ENVELOPE e^{-phi D_S^2} (Lemma 1) ->
    ``test_envelope_holds`` (headline cross-check);
  * ``ratio_R`` vs MC is a constant-factor (~1.5x) agreement -> ``test_same_order``.

Modest N_trials for speed; production figure runs use N >= 1e5.
"""
from __future__ import annotations

import math

import pytest

from src import analytic, montecarlo
from src.config import load_params

N_TRIALS = 4000   # dev value; production figures use N >= 1e5
SEED = 42
K_STDERR = 3.0    # allow +k*stderr slack on the one-directional bound


@pytest.fixture(scope="module")
def params() -> dict:
    return load_params()


def _params_at(params: dict, phiDs2: float) -> tuple[dict, float, int]:
    """Copy of params with D_S set so phi*D_S^2 = phiDs2; returns (p, delta, ell_max)."""
    p = dict(params)
    p["D_S"] = math.sqrt(phiDs2 / p["phi"])
    delta = p["D_S"] / 2.0
    ell_max = int(p["phase2_far_probe_segments"]) // 2  # far-probe neighbours per side
    return p, delta, ell_max


# ---------------------------------------------------------------------------
# 1. reproducibility
# ---------------------------------------------------------------------------
def test_mc_runs_deterministic(params: dict) -> None:
    p, delta, _ = _params_at(params, 3.0)
    r1 = montecarlo.ratio_mc(p, delta, n_trials=1500, seed=SEED)
    r2 = montecarlo.ratio_mc(p, delta, n_trials=1500, seed=SEED)
    assert r1 == r2


# ---------------------------------------------------------------------------
# 2. HEADLINE: MC ratio sits under the rigorous envelope e^{-phi D_S^2} (Lemma 1)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("phiDs2", [1.0, 2.0, 3.0, 4.0])
def test_envelope_holds(params: dict, phiDs2: float) -> None:
    p, delta, _ = _params_at(params, phiDs2)
    res = montecarlo.ratio_mc(p, delta, n_trials=N_TRIALS, seed=SEED)
    env = analytic.R_envelope(p["phi"], p["D_S"])
    assert res["ratio"] <= env + K_STDERR * res["stderr"], (
        f"phi*D_S^2={phiDs2}: ratio_mc={res['ratio']:.3e} "
        f"(stderr={res['stderr']:.1e}) exceeds envelope={env:.3e}"
    )


# ---------------------------------------------------------------------------
# 3. constant-factor agreement: MC and analytic within ~3x, with MC > analytic.
#    WHY MC > analytic: the analytic 1/r^2 freeze inflates the adjacent (small
#    u_min) term more than the far term, so the proxy ratio_R lands ~1.5x BELOW
#    the true MC ratio. This is expected -- do NOT "fix" it (see CLAUDE.md Sec. 3).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("phiDs2", [2.0, 3.0])
def test_same_order(params: dict, phiDs2: float) -> None:
    p, delta, ell_max = _params_at(params, phiDs2)
    res = montecarlo.ratio_mc(p, delta, n_trials=N_TRIALS, seed=SEED)
    analytic_R = analytic.ratio_R(p, delta, ell_max=ell_max)
    rel = res["ratio"] / analytic_R
    assert 1.0 / 3.0 <= rel <= 3.0, f"phi*D_S^2={phiDs2}: MC/analytic={rel:.2f} not in [1/3, 3]"
    assert res["ratio"] > analytic_R  # MC above analytic (freeze direction)


# ---------------------------------------------------------------------------
# 4. adjacency thesis: at the gate (phi*D_S^2 >= 3) the true ratio is small
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("phiDs2", [3.0, 4.0])
def test_far_negligible_at_gate(params: dict, phiDs2: float) -> None:
    p, delta, _ = _params_at(params, phiDs2)
    res = montecarlo.ratio_mc(p, delta, n_trials=N_TRIALS, seed=SEED)
    assert res["ratio"] <= 0.1
