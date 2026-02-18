"""Statevector VQE simulator with NaN-guarded gate operations.

Provides the hardware-efficient VQE ansatz (RY-RZ + CNOT ring) and
expectation value computation.
"""

from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# Primitive gates
# ---------------------------------------------------------------------------

CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex128)


def _renorm(psi: np.ndarray) -> np.ndarray:
    """Renormalise a statevector; reset to uniform if degenerate."""
    nrm = float(np.vdot(psi, psi).real)
    if (not np.isfinite(nrm)) or nrm <= 0:
        psi[:] = 1.0 / math.sqrt(psi.size)
    else:
        psi /= math.sqrt(nrm)
    return psi


def _apply_1q(psi: np.ndarray, gate: np.ndarray, target: int, n: int) -> np.ndarray:
    """Apply a single-qubit gate to qubit *target* of an *n*-qubit state."""
    with np.errstate(all="ignore"):
        psi_r = psi.reshape([2] * n)
        psi_r = np.moveaxis(psi_r, target, 0)
        block = psi_r.reshape(2, -1).astype(np.complex128, copy=False)
        np.nan_to_num(block, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        out = gate @ block
        np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        psi_r = out.reshape([2] + [2] * (n - 1))
        psi = np.moveaxis(psi_r, 0, target).reshape(-1)
    return psi


def _apply_2q(psi: np.ndarray, gate4: np.ndarray, q1: int, q2: int, n: int) -> np.ndarray:
    """Apply a two-qubit gate to qubits *q1*, *q2*."""
    if q1 == q2:
        return psi
    with np.errstate(all="ignore"):
        a, b = sorted((q1, q2))
        psi_r = psi.reshape([2] * n)
        psi_r = np.moveaxis(psi_r, (a, b), (0, 1))
        block = psi_r.reshape(4, -1).astype(np.complex128, copy=False)
        np.nan_to_num(block, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        out = gate4 @ block
        np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        psi_r = out.reshape(2, 2, *psi_r.shape[2:])
        psi = np.moveaxis(psi_r, (0, 1), (a, b)).reshape(-1)
    return psi


# ---------------------------------------------------------------------------
# VQE ansatz
# ---------------------------------------------------------------------------


def vqe_state(n: int, params: np.ndarray, L: int) -> np.ndarray:
    """Build the VQE statevector for a hardware-efficient RY-RZ + CNOT-ring ansatz.

    Parameters
    ----------
    n : int
        Number of qubits.
    params : ndarray of shape (2*n*L,)
        Variational parameters [ry_0..ry_{n-1}, rz_0..rz_{n-1}] x L layers.
    L : int
        Number of ansatz layers.
    """
    K = 1 << n
    psi = np.zeros(K, dtype=np.complex128)
    psi[0] = 1.0
    for layer in range(L):
        ry = params[layer * (2 * n) : layer * (2 * n) + n]
        rz = params[layer * (2 * n) + n : (layer + 1) * (2 * n)]
        for q in range(n):
            cy, sy = math.cos(float(ry[q]) / 2), math.sin(float(ry[q]) / 2)
            RY = np.array([[cy, -sy], [sy, cy]], dtype=np.complex128)
            psi = _apply_1q(psi, RY, q, n)
            cz = np.exp(-0.5j * float(rz[q]))
            sz = np.exp(+0.5j * float(rz[q]))
            RZ = np.array([[cz, 0], [0, sz]], dtype=np.complex128)
            psi = _apply_1q(psi, RZ, q, n)
        for q in range(n):
            psi = _apply_2q(psi, CNOT, q, (q + 1) % n, n)
        psi = _renorm(psi)
    return psi


# ---------------------------------------------------------------------------
# Probability extraction and expectation values
# ---------------------------------------------------------------------------


def probs_from_state(psi: np.ndarray) -> np.ndarray:
    """Extract a normalised probability vector from a statevector."""
    probs = (psi.conj() * psi).real.astype(np.float64)
    s = float(np.sum(probs))
    if (not np.isfinite(s)) or s <= 0:
        probs[:] = 1.0 / probs.size
    else:
        probs /= s
    return np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)


def expect_J(psi: np.ndarray, cut_mask: np.ndarray, w: np.ndarray):
    """Compute the expected cut value and per-edge cut probabilities.

    Returns
    -------
    J : float
        Expected cut value.
    p_cut : ndarray of shape (m,)
        Per-edge cut probabilities.
    probs : ndarray of shape (K,)
        Computational basis distribution.
    """
    probs = probs_from_state(psi)
    p_cut = probs @ cut_mask  # shape (m,)
    J = float(p_cut @ w)
    if not np.isfinite(J):
        J = 0.0
    return float(J), p_cut.astype(np.float64), probs
