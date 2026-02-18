"""Tests for paramham.simulator."""

import numpy as np

from paramham.maxcut import build_cut_mask, precompute_z
from paramham.simulator import _renorm, expect_J, probs_from_state, vqe_state


def test_vqe_state_normalized():
    n, L = 3, 2
    params = np.random.default_rng(42).uniform(-np.pi, np.pi, size=2 * n * L)
    psi = vqe_state(n, params, L)
    norm = float(np.vdot(psi, psi).real)
    assert abs(norm - 1.0) < 1e-10


def test_vqe_state_shape():
    n, L = 4, 1
    params = np.zeros(2 * n * L)
    psi = vqe_state(n, params, L)
    assert psi.shape == (16,)


def test_vqe_state_zero_params():
    """With all-zero parameters, RY(0)=I and RZ(0)=I, so only CNOT acts on |0...0>."""
    n, L = 2, 1
    params = np.zeros(2 * n * L)
    psi = vqe_state(n, params, L)
    # State should remain valid (normalized)
    norm = float(np.vdot(psi, psi).real)
    assert abs(norm - 1.0) < 1e-10


def test_probs_from_state_sum():
    psi = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    probs = probs_from_state(psi)
    assert abs(probs.sum() - 1.0) < 1e-10
    assert probs[0] == 1.0


def test_probs_from_state_degenerate():
    psi = np.zeros(4, dtype=np.complex128)
    probs = probs_from_state(psi)
    assert abs(probs.sum() - 1.0) < 1e-10
    np.testing.assert_allclose(probs, 0.25)


def test_renorm():
    psi = np.array([3.0, 4.0], dtype=np.complex128)
    psi = _renorm(psi)
    assert abs(float(np.vdot(psi, psi).real) - 1.0) < 1e-10


def test_renorm_zero():
    psi = np.zeros(4, dtype=np.complex128)
    psi = _renorm(psi)
    assert abs(float(np.vdot(psi, psi).real) - 1.0) < 1e-10


def test_expect_J():
    n = 2
    Z = precompute_z(n)
    edges = [(0, 1)]
    cm = build_cut_mask(edges, Z)
    w = np.array([1.0])
    # |01> = cut state, J should be 1.0
    psi = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.complex128)
    J, p_cut, probs = expect_J(psi, cm, w)
    assert abs(J - 1.0) < 1e-10
    assert abs(probs[1] - 1.0) < 1e-10


def test_expect_J_uncut():
    n = 2
    Z = precompute_z(n)
    edges = [(0, 1)]
    cm = build_cut_mask(edges, Z)
    w = np.array([1.0])
    # |00> = uncut state, J should be 0.0
    psi = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    J, p_cut, probs = expect_J(psi, cm, w)
    assert abs(J) < 1e-10
