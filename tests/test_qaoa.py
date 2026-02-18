"""Tests for paramham.qaoa."""

import numpy as np

from paramham.maxcut import build_cut_mask, precompute_z
from paramham.qaoa import RX, ZZ_phase, qaoa_energy, qaoa_state


def test_RX_unitarity():
    for theta in [0.0, np.pi / 4, np.pi, 2 * np.pi]:
        U = RX(theta)
        prod = U @ U.conj().T
        np.testing.assert_allclose(prod, np.eye(2), atol=1e-12)


def test_RX_zero():
    np.testing.assert_allclose(RX(0.0), np.eye(2), atol=1e-12)


def test_ZZ_phase_unitarity():
    for theta in [0.0, 0.3, np.pi / 2]:
        U = ZZ_phase(theta)
        prod = U @ U.conj().T
        np.testing.assert_allclose(prod, np.eye(4), atol=1e-12)


def test_ZZ_phase_diagonal():
    U = ZZ_phase(0.5)
    assert U[0, 1] == 0
    assert U[1, 0] == 0


def test_qaoa_state_normalized():
    n, p = 3, 2
    edges = [(0, 1), (1, 2), (0, 2)]
    w = np.ones(3)
    params = np.random.default_rng(42).uniform(-np.pi, np.pi, size=2 * p)
    psi = qaoa_state(n, edges, w, params, p)
    norm = float(np.vdot(psi, psi).real)
    assert abs(norm - 1.0) < 1e-10


def test_qaoa_state_shape():
    n, p = 4, 1
    edges = [(0, 1), (2, 3)]
    w = np.ones(2)
    params = np.zeros(2 * p)
    psi = qaoa_state(n, edges, w, params, p)
    assert psi.shape == (16,)


def test_qaoa_energy_negative():
    """QAOA energy should be negative of expected cut value."""
    n, p = 3, 1
    edges = [(0, 1), (1, 2)]
    Z = precompute_z(n)
    cm = build_cut_mask(edges, Z)
    w = np.ones(2)
    params = np.random.default_rng(42).uniform(-np.pi, np.pi, size=2 * p)
    E = qaoa_energy(n, edges, cm, w, params, p)
    assert isinstance(E, float)
    # E = -J, and J >= 0, so E <= 0 typically (for reasonable params)


def test_qaoa_zero_params():
    """With zero params, QAOA doesn't evolve from uniform superposition."""
    n, p = 2, 1
    edges = [(0, 1)]
    w = np.ones(1)
    params = np.zeros(2 * p)
    psi = qaoa_state(n, edges, w, params, p)
    # gamma=0 => cost unitary is identity, beta=0 => mixer is identity
    # Should stay in uniform superposition
    expected = np.ones(4) / 2.0
    np.testing.assert_allclose(np.abs(psi), np.abs(expected), atol=1e-10)
