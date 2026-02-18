"""Tests for paramham.families."""

import numpy as np
import pytest

from paramham.families import Family1D, FamilyEdgeWise


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestFamily1D:
    @pytest.mark.parametrize("kind", ["linear", "quadratic", "periodic"])
    def test_w_shape(self, rng, kind):
        fam = Family1D(5, kind, (0.0, 1.0), rng)
        w = fam.w(0.5)
        assert w.shape == (5,)

    @pytest.mark.parametrize("kind", ["linear", "quadratic", "periodic"])
    def test_dw_dlam_shape(self, rng, kind):
        fam = Family1D(5, kind, (0.0, 1.0), rng)
        dw = fam.dw_dlam(0.5)
        assert dw.shape == (5,)

    def test_x_range(self, rng):
        fam = Family1D(3, "linear", (0.0, 1.0), rng)
        assert fam.x(0.0) == pytest.approx(-1.0)
        assert fam.x(1.0) == pytest.approx(1.0)
        assert fam.x(0.5) == pytest.approx(0.0)

    def test_safety_bounds_positive_weights(self, rng):
        fam = Family1D(10, "linear", (0.0, 1.0), rng, safety_bounds=True)
        for lam in np.linspace(0.0, 1.0, 50):
            w = fam.w(lam)
            assert np.all(w >= 0), f"Negative weight at lam={lam}"

    def test_no_safety_bounds(self, rng):
        # Without safety bounds, weights CAN potentially go negative
        fam = Family1D(10, "linear", (0.0, 1.0), rng, safety_bounds=False)
        assert fam.A is not None  # just verify construction works

    def test_invalid_kind(self, rng):
        with pytest.raises(ValueError):
            Family1D(3, "invalid", (0.0, 1.0), rng)

    def test_gradient_finite_difference(self, rng):
        fam = Family1D(4, "linear", (0.0, 1.0), rng)
        lam = 0.5
        eps = 1e-6
        dw_analytic = fam.dw_dlam(lam)
        dw_fd = (fam.w(lam + eps) - fam.w(lam - eps)) / (2 * eps)
        np.testing.assert_allclose(dw_analytic, dw_fd, atol=1e-5)

    @pytest.mark.parametrize("kind", ["quadratic", "periodic"])
    def test_gradient_fd_other_kinds(self, rng, kind):
        fam = Family1D(4, kind, (0.0, 1.0), rng)
        lam = 0.3
        eps = 1e-6
        dw_analytic = fam.dw_dlam(lam)
        dw_fd = (fam.w(lam + eps) - fam.w(lam - eps)) / (2 * eps)
        np.testing.assert_allclose(dw_analytic, dw_fd, atol=1e-4)


class TestFamilyEdgeWise:
    @pytest.mark.parametrize("kind", ["linear", "quadratic", "periodic"])
    def test_w_shape(self, rng, kind):
        fam = FamilyEdgeWise(5, kind, (0.0, 1.0), rng)
        lam_vec = np.full(5, 0.5)
        w = fam.w(lam_vec)
        assert w.shape == (5,)

    @pytest.mark.parametrize("kind", ["linear", "quadratic", "periodic"])
    def test_dw_dlam_shape(self, rng, kind):
        fam = FamilyEdgeWise(5, kind, (0.0, 1.0), rng)
        lam_vec = np.full(5, 0.5)
        dw = fam.dw_dlam(lam_vec)
        assert dw.shape == (5,)

    def test_w_max(self, rng):
        fam = FamilyEdgeWise(5, "linear", (0.0, 1.0), rng)
        wmax = fam.w_max()
        assert wmax.shape == (5,)
        assert np.all(wmax > 0)

    def test_invalid_kind(self, rng):
        with pytest.raises(ValueError):
            FamilyEdgeWise(3, "invalid", (0.0, 1.0), rng)

    def test_gradient_finite_difference(self, rng):
        fam = FamilyEdgeWise(4, "linear", (0.0, 1.0), rng)
        lam_vec = np.array([0.3, 0.5, 0.7, 0.2])
        eps = 1e-6
        dw_analytic = fam.dw_dlam(lam_vec)
        dw_fd = (fam.w(lam_vec + eps) - fam.w(lam_vec - eps)) / (2 * eps)
        np.testing.assert_allclose(dw_analytic, dw_fd, atol=1e-5)
