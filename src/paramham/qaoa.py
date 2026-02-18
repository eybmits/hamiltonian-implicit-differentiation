"""QAOA ansatz for Max-Cut (from experiment 7).

Provides the problem-dependent QAOA circuit: cost unitary via ZZ-phase
gates and X-mixer, followed by expectation value computation.
"""

from __future__ import annotations

import math

import numpy as np

from paramham.simulator import _apply_1q, _apply_2q, _renorm, expect_J


def RX(theta: float) -> np.ndarray:
    """Single-qubit Rx rotation matrix."""
    ct = math.cos(theta / 2)
    st = math.sin(theta / 2)
    return np.array([[ct, -1j * st], [-1j * st, ct]], dtype=np.complex128)


def ZZ_phase(theta: float) -> np.ndarray:
    """Two-qubit ZZ phase gate: exp(i theta Z x Z)."""
    p = np.exp(1j * theta)
    m = np.exp(-1j * theta)
    return np.diag([p, m, m, p]).astype(np.complex128)


def qaoa_state(n: int, edges, w: np.ndarray, params: np.ndarray, p: int) -> np.ndarray:
    """Build the QAOA statevector.

    Parameters
    ----------
    n : int
        Number of qubits.
    edges : list of (int, int)
        Graph edges.
    w : ndarray of shape (m,)
        Edge weights.
    params : ndarray of shape (2*p,)
        ``[gamma_1..gamma_p, beta_1..beta_p]``.
    p : int
        Number of QAOA rounds.
    """
    gammas = params[:p]
    betas = params[p : 2 * p]

    K = 1 << n
    psi = np.ones(K, dtype=np.complex128) / math.sqrt(K)

    for layer in range(p):
        gamma = float(gammas[layer])
        beta = float(betas[layer])

        for (i, j), wij in zip(edges, w):
            theta = 0.5 * gamma * float(wij)
            psi = _apply_2q(psi, ZZ_phase(theta), int(i), int(j), n)

        gate = RX(2.0 * beta)
        for q in range(n):
            psi = _apply_1q(psi, gate, q, n)

        psi = _renorm(psi)

    return psi


def qaoa_energy(n: int, edges, cut_mask: np.ndarray, w: np.ndarray, params: np.ndarray, p: int) -> float:
    """Compute the negative expected cut value for QAOA (for minimisation)."""
    psi = qaoa_state(n, edges, w, params, p)
    J, _, _ = expect_J(psi, cut_mask, w)
    return -J
