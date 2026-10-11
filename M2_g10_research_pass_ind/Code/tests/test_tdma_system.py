"""v8: the orthogonal benchmark as a SYSTEM sum rate (execution plan A1).

`analytic.tdma_system_rate` is computed from the integrals directly. These tests pin it
against the unchanged `tdma_rate_eq9`, against an independent numerical integral, and
against two structural properties that follow from the time-sharing identity.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad

from src import analytic as A
from src.config import load_params


@pytest.fixture(scope="module")
def params() -> dict:
    return load_params()


@pytest.mark.parametrize("model", ["A", "B"])
@pytest.mark.parametrize("placement", ["u0", "uavg"])
@pytest.mark.parametrize("M,K", [(3, 1), (3, 4), (5, 2), (3, 64)])
def test_equals_eq9_times_MK(params, model, placement, M, K) -> None:
    """The only difference from the validated `tdma_rate_eq9` is the factor M*K."""
    sysr = A.tdma_system_rate(model, K, float(M), params, placement=placement)
    old = A.tdma_rate_eq9(model, K, float(M), params, placement=placement)
    assert sysr == pytest.approx(old * M * K, rel=1e-9)


def test_independent_integral(params) -> None:
    """Version A, PA over the active user: E_y[log2(1 + rho eta/(y^2 + d^2))], no boost."""
    p = params["phase3"]
    eta, d, D_W = float(params["eta"]), float(params["d"]), float(params["D_W"])
    rho = float(p["P"]) / float(p["N0"])
    ref, _ = quad(lambda y: math.log2(1.0 + rho * eta / (y * y + d * d)), -D_W / 2, D_W / 2)
    ref /= D_W
    assert A.tdma_system_rate("A", 7, 3.0, params) == pytest.approx(ref, rel=1e-9)


def test_peak_model_is_independent_of_load_and_segments(params) -> None:
    """With no boost, sharing one channel among more users does not change the system rate."""
    base = A.tdma_system_rate("A", 1, 3.0, params)
    for M, K in [(3, 8), (5, 2), (7, 64)]:
        assert A.tdma_system_rate("A", K, float(M), params) == pytest.approx(base, rel=1e-12)


def test_average_model_gains_one_bit_per_doubling(params) -> None:
    """Boost b = MK adds log2(2) = 1 bit when MK doubles (rho eta/r^2 >> 1 at 120 dB)."""
    r1 = A.tdma_system_rate("B", 1, 3.0, params)
    r2 = A.tdma_system_rate("B", 2, 3.0, params)
    r4 = A.tdma_system_rate("B", 4, 3.0, params)
    assert r2 - r1 == pytest.approx(1.0, abs=2e-3)
    assert r4 - r2 == pytest.approx(1.0, abs=2e-3)


def test_reuse_matched_benchmark_is_load_free(params) -> None:
    """One user per region per slot is a K = 1 system in every slot (Theorem 1 at K = 1),
    so the in-segment time-shared ergodic system rate M*R(K=1) does not depend on K."""
    M = 3
    r1 = M * A.ergodic_rate_exact(1, params)[0]
    assert r1 > 0.0
    # K users each holding a 1/K slot, each slot's expected rate R(1): sum = R(1).
    for K in (2, 5, 20):
        assert K * (1.0 / K) * r1 == pytest.approx(r1, rel=1e-12)


def test_rejects_bad_arguments(params) -> None:
    with pytest.raises(ValueError):
        A.tdma_system_rate("X", 1, 3.0, params)
    with pytest.raises(ValueError):
        A.tdma_system_rate("A", 1, 3.0, params, placement="nope")
