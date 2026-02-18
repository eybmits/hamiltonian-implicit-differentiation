"""Tests for paramham.metrics."""

import numpy as np

from paramham.metrics import mean_stderr, step_auc, step_interp, tail_probability, win_rate


class TestStepInterp:
    def test_basic(self):
        evals = np.array([1.0, 3.0, 5.0])
        values = np.array([10.0, 20.0, 30.0])
        budgets = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        result = step_interp(evals, values, budgets)
        np.testing.assert_array_equal(result, [0.0, 10.0, 10.0, 20.0, 20.0, 30.0])

    def test_anchor_value(self):
        evals = np.array([2.0])
        values = np.array([5.0])
        budgets = np.array([0.0, 1.0, 2.0, 3.0])
        result = step_interp(evals, values, budgets, anchor_value=-1.0)
        np.testing.assert_array_equal(result, [-1.0, -1.0, 5.0, 5.0])

    def test_empty(self):
        result = step_interp(np.array([]), np.array([]), np.array([1.0, 2.0]), anchor_value=0.0)
        np.testing.assert_array_equal(result, [0.0, 0.0])

    def test_sort(self):
        evals = np.array([3.0, 1.0])
        values = np.array([30.0, 10.0])
        budgets = np.array([0.0, 1.0, 2.0, 3.0])
        result = step_interp(evals, values, budgets, sort=True)
        np.testing.assert_array_equal(result, [0.0, 10.0, 10.0, 30.0])


class TestStepAUC:
    def test_constant(self):
        evals = np.array([0.0])
        values = np.array([5.0])
        auc = step_auc(evals, values, 10.0)
        assert abs(auc - 5.0) < 1e-10

    def test_zero_budget(self):
        assert step_auc(np.array([1.0]), np.array([5.0]), 0.0) == 0.0

    def test_empty(self):
        assert step_auc(np.array([]), np.array([]), 10.0) == 0.0

    def test_step(self):
        evals = np.array([0.0, 5.0])
        values = np.array([2.0, 4.0])
        auc = step_auc(evals, values, 10.0)
        # Area: 2*5 + 4*5 = 30, normalised: 30/10 = 3.0
        assert abs(auc - 3.0) < 1e-10


class TestMeanStderr:
    def test_basic(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mu, se = mean_stderr(x)
        assert abs(mu - 3.0) < 1e-10
        assert se > 0

    def test_2d(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        mu, se = mean_stderr(x, axis=0)
        np.testing.assert_allclose(mu, [3.0, 4.0])
        assert se.shape == (2,)


class TestWinRate:
    def test_all_positive(self):
        assert win_rate(np.array([1.0, 2.0, 3.0])) == 1.0

    def test_all_negative(self):
        assert win_rate(np.array([-1.0, -2.0])) == 0.0

    def test_mixed(self):
        assert abs(win_rate(np.array([1.0, -1.0])) - 0.5) < 1e-10

    def test_empty(self):
        assert np.isnan(win_rate(np.array([])))

    def test_with_nan(self):
        assert win_rate(np.array([1.0, float("nan"), 1.0])) == 1.0


class TestTailProbability:
    def test_hit_metric(self):
        probs = np.array([0.0, 0.5, 0.3, 0.2])
        cut_vals = np.array([0.0, 5.0, 3.0, 4.0])
        p, thr = tail_probability(probs, cut_vals, metric="hit", J_star=5.0, hit_eps=0.01)
        assert 0.0 <= p <= 1.0
        assert thr > 0

    def test_topk_metric(self):
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        cut_vals = np.array([1.0, 2.0, 3.0, 4.0])
        p, thr = tail_probability(probs, cut_vals, metric="topk", J_star=4.0)
        assert 0.0 <= p <= 1.0

    def test_unknown_metric(self):
        probs = np.array([1.0])
        cut_vals = np.array([1.0])
        p, thr = tail_probability(probs, cut_vals, metric="unknown", J_star=1.0)
        assert np.isnan(p)
        assert np.isnan(thr)
