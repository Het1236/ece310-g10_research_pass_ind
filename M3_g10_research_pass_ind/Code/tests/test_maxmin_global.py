"""Gates for src/maxmin_global.py (certified-global max-min PA placement, Proposition 6 /
app:ao). Reproduces the two reference instances validated in this-session's convergence
track (recorded in CLAUDE.md's dated status-log entry and paper/sections/appendix.tex
Appendix F): a pinned instance where V(x) is unimodal, and a disparity instance where
V(x) is genuinely multimodal and the naive AO alternation gets trapped in the secondary
local maximum.
"""
from __future__ import annotations

import numpy as np
import pytest

from src import analytic, maxmin_global
from src.config import load_params

_TOL = 1e-3          # Piyavskii-Shubert certified-gap tolerance used throughout
_VAL_TOL = 5e-3      # value-match tolerance against the recorded reference V*


@pytest.fixture(scope="module")
def params() -> dict:
    return load_params()


@pytest.fixture(scope="module")
def env(params) -> dict:
    D_S = float(params["D_S"])
    K = 2
    floor = (float(params["phase3"]["N0"])
             + 2.0 * K * float(params["phase3"]["P"]) * analytic.L1_exact(params))
    return {
        "D_S": D_S, "d": float(params["d"]), "eta": float(params["eta"]),
        "P_max": float(params["phase4"]["P_max"]), "floor": floor,
    }


def test_pinned_instance_matches_reference(env):
    """ux=[12.416,10.171], uy=[-1.790,-3.107]: recorded reference V*=2.579661 at
    x*=12.415, unimodal, gap<1e-3."""
    ux = np.array([12.416, 10.171])
    uy = np.array([-1.790, -3.107])
    sol = maxmin_global.maxmin_global_placement(
        ux, uy, 0.0, env["D_S"], env["d"], env["eta"], env["floor"], "A",
        P_max=env["P_max"], tol=_TOL)
    assert sol["gap"] <= _TOL + 1e-9, f"certified gap {sol['gap']} exceeds tol"
    assert abs(sol["f"] - 2.579661) <= _VAL_TOL, f"V* {sol['f']} != reference 2.579661"
    assert abs(sol["x"] - 12.415) <= 0.05, f"x* {sol['x']} != reference 12.415"


def test_disparity_instance_multimodal_and_matches_reference(env):
    """ux=[4.0,9.0], uy=[0.3,4.6]: recorded reference V*=2.77412 at x*=3.999, with a
    breakpoint near x=8.607 separating the global max (x~4.0) from a secondary local
    max (x~9.0, V~2.00) -- the genuinely multimodal instance that traps naive AO."""
    ux = np.array([4.0, 9.0])
    uy = np.array([0.3, 4.6])
    sol = maxmin_global.maxmin_global_placement(
        ux, uy, 0.0, env["D_S"], env["d"], env["eta"], env["floor"], "A",
        P_max=env["P_max"], tol=_TOL)
    assert sol["gap"] <= _TOL + 1e-9
    assert abs(sol["f"] - 2.77412) <= _VAL_TOL, f"V* {sol['f']} != reference 2.77412"
    assert abs(sol["x"] - 3.999) <= 0.05, f"x* {sol['x']} != reference 3.999"
    assert len(sol["breakpoints"]) >= 1, "expected >=1 SIC breakpoint for a K=2 instance"


def test_deterministic_variance_zero(env):
    """Repeated calls with identical inputs return bit-identical results (no random
    initialisation anywhere in the search) -- the property that AO's multi-start lacks."""
    ux = np.array([4.0, 9.0])
    uy = np.array([0.3, 4.6])
    vals = [
        maxmin_global.maxmin_global_placement(
            ux, uy, 0.0, env["D_S"], env["d"], env["eta"], env["floor"], "A",
            P_max=env["P_max"], tol=_TOL)["f"]
        for _ in range(5)
    ]
    assert max(vals) - min(vals) == 0.0, f"non-deterministic: spread {max(vals)-min(vals)}"


def test_sic_breakpoints_closed_form_root():
    """A single-crossing two-user instance: breakpoint is the exact midpoint in x when
    the two users have identical y (symmetric C_i=C_j collapses the closed-form root to
    the arithmetic mean of the two x-anchors)."""
    ux = np.array([2.0, 8.0])
    uy = np.array([1.0, 1.0])
    bps = maxmin_global.sic_breakpoints(ux, uy, d=0.5, lo=0.0, hi=10.0)
    assert len(bps) == 1
    assert abs(bps[0] - 5.0) <= 1e-9
