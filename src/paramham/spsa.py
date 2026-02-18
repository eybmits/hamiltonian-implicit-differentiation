"""SPSA (Simultaneous Perturbation Stochastic Approximation) optimiser.

NaN-guarded variant: gracefully handles non-finite energy evaluations.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from paramham.seeds import to_uint_seed


def spsa_minimize(
    energy_fun,
    p0: np.ndarray,
    bounds: List[Tuple[float, float]],
    iters: int,
    seed: int,
    a: float = 0.2,
    c: float = 0.12,
    A: float = 20.0,
    alpha: float = 0.602,
    gamma: float = 0.101,
) -> Tuple[np.ndarray, float, int]:
    """Minimise *energy_fun* via SPSA.

    Returns ``(best_params, best_energy, num_evals)``.
    Each iteration uses 3 energy evaluations (E+, E-, E_center).
    """
    rng = np.random.default_rng(to_uint_seed(seed))
    p = np.asarray(p0, dtype=np.float64).copy()
    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)

    best_p = p.copy()
    best_E = float("inf")
    evals = 0

    for k in range(1, iters + 1):
        ak = a / ((k + A) ** alpha)
        ck = c / (k**gamma)
        delta = rng.choice([-1.0, 1.0], size=p.size)

        Ep = float(energy_fun(np.clip(p + ck * delta, lo, hi)))
        Em = float(energy_fun(np.clip(p - ck * delta, lo, hi)))
        evals += 2

        if (not np.isfinite(Ep)) or (not np.isfinite(Em)) or ck <= 0:
            ghat = np.zeros_like(p)
        else:
            ghat = (Ep - Em) / (2.0 * ck) * delta

        p = np.clip(p - ak * ghat, lo, hi)

        E = float(energy_fun(p))
        evals += 1
        if np.isfinite(E) and E < best_E:
            best_E = E
            best_p = p.copy()

    if not np.isfinite(best_E):
        best_E = float(energy_fun(best_p))
        evals += 1

    return best_p, best_E, evals
