"""Z-basis precomputation, cut masks, and classical Max-Cut diagnostics."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def precompute_z(n: int) -> np.ndarray:
    """Z eigenvalues for each qubit and computational basis state.

    Returns Z of shape ``(n, 2**n)`` with entries in {+1, -1},
    using big-endian qubit ordering.
    """
    K = 1 << n
    idx = np.arange(K, dtype=np.uint32)
    Z = np.empty((n, K), dtype=np.int8)
    for q in range(n):
        bitpos = n - 1 - q
        Z[q] = 1 - 2 * ((idx >> bitpos) & 1).astype(np.int8)
    return Z


def build_cut_mask(edges: List[Tuple[int, int]], Z: np.ndarray) -> np.ndarray:
    """Build the cut indicator matrix.

    Returns ``cut_mask`` of shape ``(2**n, m)`` where
    ``cut_mask[x, e] = (1 - Z_i Z_j) / 2`` for edge ``e = (i, j)``.
    """
    K = Z.shape[1]
    m = len(edges)
    cut = np.empty((K, m), dtype=np.float64)
    for e, (i, j) in enumerate(edges):
        cut[:, e] = 0.5 * (1.0 - (Z[i] * Z[j]).astype(np.float64))
    return cut


def build_ZZ_edges(edges: List[Tuple[int, int]], Z: np.ndarray) -> np.ndarray:
    """Precompute ZZ correlators per edge.

    Returns ``ZZ`` of shape ``(m, 2**n)`` with entries in {+1, -1}.
    """
    K = Z.shape[1]
    m = len(edges)
    ZZ = np.empty((m, K), dtype=np.int8)
    for e, (i, j) in enumerate(edges):
        ZZ[e] = (Z[i] * Z[j]).astype(np.int8)
    return ZZ


def classical_Jstar(fam, cut_mask: np.ndarray, grid_points: int) -> Tuple[float, float]:
    """Compute J* = max_{lambda, z} J(z; lambda) on a lambda grid.

    Parameters
    ----------
    fam : Family1D
        A parametric weight family with ``.w(lam)``, ``.lam_min``, ``.lam_max``.
    cut_mask : ndarray of shape (K, m)
    grid_points : int

    Returns
    -------
    J_star : float
        The maximum classical cut value.
    lam_star : float
        The grid lambda at which the maximum is attained.
    """
    lams = np.linspace(fam.lam_min, fam.lam_max, int(grid_points))
    bestJ = -1e30
    bestLam = float(lams[0])
    for lam in lams:
        w = fam.w(float(lam)).astype(np.float64)
        with np.errstate(all="ignore"):
            vals = cut_mask @ w
        vals = np.nan_to_num(vals, nan=-1e30, posinf=-1e30, neginf=-1e30)
        v = float(np.max(vals))
        if v > bestJ:
            bestJ = v
            bestLam = float(lam)
    if not np.isfinite(bestJ) or bestJ <= 0:
        bestJ = 1.0
    return float(bestJ), float(bestLam)
