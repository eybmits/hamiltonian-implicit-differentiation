"""Tests for paramham.spsa."""

import numpy as np

from paramham.spsa import spsa_minimize


def test_quadratic_minimum():
    """SPSA should find the minimum of a simple quadratic."""

    def quadratic(x):
        return float(np.sum((x - 1.0) ** 2))

    p0 = np.array([0.0, 0.0])
    bounds = [(-5.0, 5.0), (-5.0, 5.0)]
    best_p, best_E, evals = spsa_minimize(quadratic, p0, bounds, iters=200, seed=42)
    assert best_E < 0.5
    np.testing.assert_allclose(best_p, [1.0, 1.0], atol=0.5)


def test_eval_count():
    """Each iteration should use 3 evals (E+, E-, E_center)."""
    call_count = [0]

    def f(x):
        call_count[0] += 1
        return float(np.sum(x**2))

    p0 = np.array([1.0])
    bounds = [(-5.0, 5.0)]
    _, _, evals = spsa_minimize(f, p0, bounds, iters=10, seed=0)
    assert evals == call_count[0]
    assert evals == 30  # 3 per iteration


def test_nan_handling():
    """SPSA should handle NaN returns gracefully."""
    calls = [0]

    def f_with_nans(x):
        calls[0] += 1
        if calls[0] % 5 == 0:
            return float("nan")
        return float(np.sum(x**2))

    p0 = np.array([2.0, 2.0])
    bounds = [(-5.0, 5.0), (-5.0, 5.0)]
    best_p, best_E, evals = spsa_minimize(f_with_nans, p0, bounds, iters=50, seed=42)
    assert np.isfinite(best_E)


def test_respects_bounds():
    """Solution should stay within bounds."""

    def f(x):
        return float(np.sum((x - 10.0) ** 2))

    p0 = np.array([0.0])
    bounds = [(-1.0, 1.0)]
    best_p, _, _ = spsa_minimize(f, p0, bounds, iters=50, seed=42)
    assert -1.0 <= best_p[0] <= 1.0
