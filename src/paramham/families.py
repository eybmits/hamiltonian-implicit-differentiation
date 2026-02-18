"""Parametrised weight families for Max-Cut Hamiltonians.

Provides ``Family1D`` (scalar lambda) and ``FamilyEdgeWise`` (vector lambda).
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


class Family1D:
    r"""Canonical bounded 1D families: ``w_e(lambda) = wbar_e + A_e * f_e(x)``.

    *x* = 2*(lambda - mid) / Delta in [-1, 1].

    Response functions (mean-zero, RMS-normalised on Uniform[-1,1]):
      - linear:    f = sqrt(3) * s * x
      - quadratic: f = sqrt(45/4) * s * (x^2 - 1/3)
      - periodic:  f = sqrt(2) * cos(pi * k * x + phi)

    Parameters
    ----------
    m : int
        Number of edges.
    kind : str
        One of ``"linear"``, ``"quadratic"``, ``"periodic"``.
    lam_bounds : tuple of float
        ``(lam_min, lam_max)``.
    rng : numpy.random.Generator
    periodic_K : int
        Max frequency index for periodic family.
    wbar_range : tuple of float
        Range for baseline weights.
    amp_range : tuple of float
        Range for amplitudes.
    safety_bounds : bool
        If True, clip amplitudes so weights stay positive on the domain.
    """

    def __init__(
        self,
        m: int,
        kind: str,
        lam_bounds: Tuple[float, float],
        rng: np.random.Generator,
        periodic_K: int = 6,
        wbar_range: Tuple[float, float] = (2.0, 3.0),
        amp_range: Tuple[float, float] = (0.3, 0.8),
        safety_bounds: bool = True,
    ):
        self.kind = kind
        self.lam_min, self.lam_max = map(float, lam_bounds)
        self.mid = 0.5 * (self.lam_min + self.lam_max)
        self.Delta = max(1e-12, self.lam_max - self.lam_min)
        self.dx_dlam = 2.0 / self.Delta

        self.wbar = rng.uniform(*wbar_range, size=m).astype(np.float64)
        self.A = rng.uniform(*amp_range, size=m).astype(np.float64)

        if kind in ("linear", "quadratic"):
            self.s = rng.choice([-1.0, +1.0], size=m).astype(np.float64)
            self.k = None
            self.phi = None
        elif kind == "periodic":
            self.s = None
            self.k = rng.integers(1, periodic_K + 1, size=m).astype(np.float64)
            self.phi = rng.uniform(0.0, 2 * np.pi, size=m).astype(np.float64)
        else:
            raise ValueError("kind must be one of: linear, quadratic, periodic")

        if safety_bounds:
            f_min = {
                "linear": -math.sqrt(3.0),
                "quadratic": -math.sqrt(45.0 / 4.0) * (1.0 / 3.0),
                "periodic": -math.sqrt(2.0),
            }[kind]
            w_min_target = 0.05
            maxA = (self.wbar - w_min_target) / max(1e-12, -f_min)
            self.A = np.minimum(self.A, np.maximum(0.0, maxA))

    def x(self, lam: float) -> float:
        return 2.0 * (float(lam) - self.mid) / self.Delta

    def f_df(self, x: float):
        x = float(x)
        if self.kind == "linear":
            c = math.sqrt(3.0)
            f = c * self.s * x
            df = c * self.s
        elif self.kind == "quadratic":
            c = math.sqrt(45.0 / 4.0)
            f = c * self.s * (x * x - 1.0 / 3.0)
            df = c * self.s * (2.0 * x)
        else:
            c = math.sqrt(2.0)
            arg = math.pi * self.k * x + self.phi
            f = c * np.cos(arg)
            df = c * (-math.pi * self.k) * np.sin(arg)
        return f, df

    def w(self, lam: float) -> np.ndarray:
        f, _ = self.f_df(self.x(lam))
        w = self.wbar + self.A * f
        return np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)

    def dw_dlam(self, lam: float) -> np.ndarray:
        _, df = self.f_df(self.x(lam))
        dw = self.A * df * self.dx_dlam
        return np.nan_to_num(dw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)


class FamilyEdgeWise:
    r"""Edge-wise parametric coupling family where lambda is a vector.

    For each edge e: ``w_e(lambda_e) = wbar_e + A_e * f_e(x(lambda_e))``
    with x(lambda) = 2*(lambda - mid) / Delta in [-1, 1].
    """

    def __init__(
        self,
        m: int,
        kind: str,
        lam_bounds: Tuple[float, float],
        rng: np.random.Generator,
        periodic_K: int = 6,
    ):
        self.kind = str(kind)
        self.lam_min, self.lam_max = map(float, lam_bounds)
        self.mid = 0.5 * (self.lam_min + self.lam_max)
        self.Delta = max(1e-12, self.lam_max - self.lam_min)
        self.dx_dlam = 2.0 / self.Delta

        self.wbar = rng.uniform(2.0, 3.0, size=m).astype(np.float64)
        self.A = rng.uniform(0.3, 0.8, size=m).astype(np.float64)

        if self.kind in ("linear", "quadratic"):
            self.s = rng.choice([-1.0, +1.0], size=m).astype(np.float64)
            self.k = None
            self.phi = None
        elif self.kind == "periodic":
            self.s = None
            self.k = rng.integers(1, periodic_K + 1, size=m).astype(np.float64)
            self.phi = rng.uniform(0.0, 2 * np.pi, size=m).astype(np.float64)
        else:
            raise ValueError("family kind must be linear, quadratic, or periodic")

    def x(self, lam_vec: np.ndarray) -> np.ndarray:
        lam_vec = np.asarray(lam_vec, float)
        return 2.0 * (lam_vec - self.mid) / self.Delta

    def f_df(self, x: np.ndarray):
        x = np.asarray(x, float)
        if self.kind == "linear":
            c = math.sqrt(3.0)
            f = c * self.s * x
            df = c * self.s * np.ones_like(x)
        elif self.kind == "quadratic":
            c = math.sqrt(45.0 / 4.0)
            f = c * self.s * (x * x - 1.0 / 3.0)
            df = c * self.s * (2.0 * x)
        else:
            c = math.sqrt(2.0)
            arg = math.pi * self.k * x + self.phi
            f = c * np.cos(arg)
            df = c * (-math.pi * self.k) * np.sin(arg)
        return f, df

    def w(self, lam_vec: np.ndarray) -> np.ndarray:
        f, _ = self.f_df(self.x(lam_vec))
        w = self.wbar + self.A * f
        return np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)

    def dw_dlam(self, lam_vec: np.ndarray) -> np.ndarray:
        _, df = self.f_df(self.x(lam_vec))
        dw = self.A * df * self.dx_dlam
        return np.nan_to_num(dw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)

    def w_max(self, grid_points: int = 801) -> np.ndarray:
        """Per-edge maximum weight over lambda in [lam_min, lam_max]."""
        lams = np.linspace(self.lam_min, self.lam_max, int(grid_points), dtype=float)
        x = 2.0 * (lams - self.mid) / self.Delta

        if self.kind == "linear":
            c = math.sqrt(3.0)
            f_max = c * np.ones_like(self.wbar)
        elif self.kind == "quadratic":
            X = x[None, :]
            c = math.sqrt(45.0 / 4.0)
            f_grid = c * self.s[:, None] * (X * X - 1.0 / 3.0)
            f_max = np.max(f_grid, axis=1)
        else:
            X = x[None, :]
            c = math.sqrt(2.0)
            arg = math.pi * self.k[:, None] * X + self.phi[:, None]
            f_grid = c * np.cos(arg)
            f_max = np.max(f_grid, axis=1)

        wmax = self.wbar + self.A * f_max
        return np.nan_to_num(wmax, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
