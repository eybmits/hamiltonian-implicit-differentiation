#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp0A_budget_efficiency.py
=========================

Experiment 0A (NPJ Core): Budget Efficiency of ID vs Black-box FD on the *value function* F(λ)
----------------------------------------------------------------------------------------------

We study a 1D parametrized Max-Cut Hamiltonian family

    H(λ) = Σ_e w_e(λ) C_e ,     C_e = (I - Z_i Z_j)/2 ,

and the VQE value function (bilevel target)

    F(λ) = max_θ  J(θ,λ) ,      J(θ,λ) = ⟨H(λ)⟩_θ = Σ_e w_e(λ) p_cut,e(θ) ,
                               p_cut,e = (1 - ⟨Z_i Z_j⟩)/2 .

Outer-loop signals compared under a *matched outer update rule* (same step-size schedule):
  - ID (correlator reuse / implicit differentiation / envelope principle):
        g_ID(λ_t) = ∂_λ J(θ*(λ_t), λ_t) = Σ_e w'_e(λ_t) p_cut,e
    computed after ONE inner solve at λ_t, reusing the same ZZ correlators measured for J.

  - BB-FD (black-box finite differences on F):
        g_FD(λ_t) ≈ [F(λ_t+c) - F(λ_t-c)]/(2c)
    where each query F(·) requires an additional inner solve at λ±c.
    => ~2 extra inner solves per outer step (the bilevel cost model).

Canonical response families (mean-zero, RMS-normalized on x~Unif[-1,1]):
    x = 2*(λ-mid)/Δ ∈ [-1,1]
    w_e(λ) = w̄_e + A_e f_e(x)

    f_lin  = √3 s x
    f_quad = √(45/4) s (x^2 - 1/3)
    f_per  = √2 cos(π k x + φ)

What this script produces
-------------------------
For each family kind in {linear, quadratic, periodic} and for shots in {exact, S}:
  - runs multiple random graph instances (seeds)
  - runs ID and BB-FD (re-solve) with identical hyperparameters
  - reports mean ± stderr best-so-far vs cumulative energy evaluations (budget axis)
  - writes a compact table (CSV + LaTeX) at a fixed evaluation budget B

Outputs are written to --out.

Quick run (small):
  python exp0A_budget_efficiency.py --num_seeds 3 --outer 10 --inner 10 --n 10 --shots_list 0,128

Paper-ish run:
  python exp0A_budget_efficiency.py --num_seeds 20 --outer 30 --inner 28 --n 12 --shots_list 0,256 --budget 2500

Notes
-----
- "Energy evaluation" means one call to the (possibly shot-noisy) estimator of J(θ,λ).
  With fixed shots per evaluation, this is the right cost unit for budget plots.
- For n>~14, exact statevector simulation becomes heavy; defaults are chosen for n<=12.
"""

from __future__ import annotations

import argparse
import csv
import math
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# -----------------------------------------------------------------------------
# Silence noisy-but-harmless PDF font timestamp chatter (matplotlib->fontTools)
# -----------------------------------------------------------------------------
warnings.filterwarnings("ignore", message=".*timestamp seems very low.*")
warnings.filterwarnings("ignore", message=".*regarding as unix timestamp.*")
_ft = logging.getLogger("fontTools")
_ft.setLevel(logging.ERROR)
_ft.propagate = False
if not _ft.handlers:
    _ft.addHandler(logging.NullHandler())


# =============================================================================
# 1) Pub-ish plotting
# =============================================================================

COLORS = {
    "ID": "#D62728",   # red
    "FD": "#1F77B4",   # blue
    "GT": "#000000",   # black
}

COL_W = 3.37
FULL_W = 6.95
H = 2.7


def set_pub_style(grid: bool = False, base_size: int = 8):
    mpl.rcdefaults()
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Liberation Serif"],
        "font.size": base_size,
        "axes.labelsize": base_size + 1,
        "legend.fontsize": base_size - 1,
        "xtick.labelsize": base_size,
        "ytick.labelsize": base_size,
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "lines.linewidth": 1.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.grid": grid,
        "grid.alpha": 0.15,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "savefig.transparent": False,   # avoid PDF alpha artefacts
        "figure.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def _savefig(fig: plt.Figure, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".pdf"]:
        fig.savefig(path, dpi=600)
    else:
        fig.savefig(path)


# =============================================================================
# 2) Utilities
# =============================================================================

def to_uint_seed(seed: int) -> int:
    return int(seed) % (2**32 - 1)


def stderr(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size <= 1:
        return float("nan")
    return float(np.std(x, ddof=1) / math.sqrt(x.size))


def step_interp(evals: np.ndarray, values: np.ndarray, budgets: np.ndarray) -> np.ndarray:
    """
    Piecewise-constant interpolation:
      returns v(b) = values[last index with evals<=b], with v=0 for b < evals[0].
    Assumes evals is strictly increasing (or non-decreasing).
    """
    evals = np.asarray(evals, dtype=float)
    values = np.asarray(values, dtype=float)
    budgets = np.asarray(budgets, dtype=float)

    # Ensure a (0,0) anchor
    if evals.size == 0:
        return np.zeros_like(budgets)
    if evals[0] > 0:
        evals = np.concatenate([[0.0], evals])
        values = np.concatenate([[0.0], values])

    idx = np.searchsorted(evals, budgets, side="right") - 1
    idx = np.clip(idx, 0, values.size - 1)
    return values[idx]


def auc_trapz(budgets: np.ndarray, values: np.ndarray) -> float:
    budgets = np.asarray(budgets, dtype=float)
    values = np.asarray(values, dtype=float)
    if budgets.size < 2:
        return float("nan")
    return float(np.trapz(values, budgets))


# =============================================================================
# 3) Graph + classical patterns
# =============================================================================

def generate_random_graph(n: int, p_edge: float, rng: np.random.Generator) -> List[Tuple[int, int]]:
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p_edge:
                edges.append((i, j))
    return edges


def precompute_z_big_endian(n: int) -> np.ndarray:
    """Z patterns: shape (n, 2^n), ±1, big-endian qubit->bit mapping."""
    K = 1 << n
    idx = np.arange(K, dtype=np.uint32)
    Z = np.empty((n, K), dtype=np.int8)
    for q in range(n):
        bitpos = n - 1 - q
        Z[q] = 1 - 2 * ((idx >> bitpos) & 1).astype(np.int8)
    return Z


def precompute_ZZ_and_cutmask(edges: List[Tuple[int, int]], Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      ZZ: shape (m, K) with entries ±1 (Z_i Z_j per edge)
      cut_mask: shape (K, m) with entries 0/1 (edge cut indicator)
    """
    m = len(edges)
    K = Z.shape[1]
    ZZ = np.empty((m, K), dtype=np.int8)
    for e, (i, j) in enumerate(edges):
        ZZ[e] = (Z[i] * Z[j]).astype(np.int8)
    cut_mask = 0.5 * (1.0 - ZZ.astype(np.float64)).T  # (K,m)
    return ZZ, cut_mask


# =============================================================================
# 4) Canonical 1D families (mean-zero, RMS=1 on x~Unif[-1,1])
# =============================================================================

class Family1D:
    def __init__(self, m: int, kind: str, lam_bounds: Tuple[float, float],
                 rng: np.random.Generator, periodic_K: int = 6,
                 wbar_range: Tuple[float, float] = (2.0, 3.0),
                 amp_range: Tuple[float, float] = (0.3, 0.8)):
        self.kind = kind
        self.lam_min, self.lam_max = map(float, lam_bounds)
        self.mid = 0.5 * (self.lam_min + self.lam_max)
        self.Delta = max(1e-12, self.lam_max - self.lam_min)
        self.dx_dlam = 2.0 / self.Delta

        self.wbar = rng.uniform(*wbar_range, size=m).astype(np.float64)
        self.A = rng.uniform(*amp_range, size=m).astype(np.float64)

        if kind in ("linear", "quadratic"):
            self.s = rng.choice([-1.0, +1.0], size=m).astype(np.float64)
        elif kind == "periodic":
            self.k = rng.integers(1, periodic_K + 1, size=m).astype(np.float64)
            self.phi = rng.uniform(0.0, 2*np.pi, size=m).astype(np.float64)
        else:
            raise ValueError("kind must be one of: linear, quadratic, periodic")

        # Safety: ensure weights stay positive on the whole domain (conservative bound)
        f_min = {
            "linear": -math.sqrt(3.0),
            "quadratic": -math.sqrt(45.0/4.0) * (1.0/3.0),  # x^2-1/3 min = -1/3
            "periodic": -math.sqrt(2.0),
        }[kind]
        w_min_target = 0.05
        # w = wbar + A*f  >= w_min_target   for worst-case f=f_min (<0)
        maxA = (self.wbar - w_min_target) / max(1e-12, -f_min)
        self.A = np.minimum(self.A, np.maximum(0.0, maxA))

    def x(self, lam: float) -> float:
        return 2.0 * (float(lam) - self.mid) / self.Delta

    def f_df_dx(self, x: float) -> Tuple[np.ndarray, np.ndarray]:
        x = float(x)
        if self.kind == "linear":
            c = math.sqrt(3.0)
            f = c * self.s * x
            df = c * self.s
        elif self.kind == "quadratic":
            c = math.sqrt(45.0/4.0)
            f = c * self.s * (x*x - 1.0/3.0)
            df = c * self.s * (2.0 * x)
        else:
            c = math.sqrt(2.0)
            arg = math.pi * self.k * x + self.phi
            f = c * np.cos(arg)
            df = c * (-math.pi * self.k) * np.sin(arg)
        return f, df

    def w(self, lam: float) -> np.ndarray:
        f, _ = self.f_df_dx(self.x(lam))
        w = self.wbar + self.A * f
        return np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)

    def dw_dlam(self, lam: float) -> np.ndarray:
        _, df = self.f_df_dx(self.x(lam))
        dw = self.A * df * self.dx_dlam
        return np.nan_to_num(dw, nan=0.0, posinf=0.0, neginf=0.0)


# =============================================================================
# 5) VQE ansatz (exact statevector) + (optional) shot estimator
# =============================================================================

CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]], dtype=np.complex128)


def _renorm_state(psi: np.ndarray) -> np.ndarray:
    nrm = float(np.vdot(psi, psi).real)
    if (not np.isfinite(nrm)) or nrm <= 0:
        psi[:] = 1.0 / math.sqrt(psi.size)
    else:
        psi /= math.sqrt(nrm)
    # wipe NaNs/Infs if any
    np.nan_to_num(psi, copy=False)
    return psi


def _apply_1q(psi: np.ndarray, gate: np.ndarray, target: int, n: int) -> np.ndarray:
    psi_r = psi.reshape([2] * n)
    psi_r = np.moveaxis(psi_r, target, 0)
    block = psi_r.reshape(2, -1)
    out = gate @ block
    psi_r = out.reshape([2] + [2] * (n - 1))
    psi = np.moveaxis(psi_r, 0, target).reshape(-1)
    return psi


def _apply_2q(psi: np.ndarray, gate4: np.ndarray, q1: int, q2: int, n: int) -> np.ndarray:
    if q1 == q2:
        return psi
    a, b = sorted((q1, q2))
    psi_r = psi.reshape([2] * n)
    psi_r = np.moveaxis(psi_r, (a, b), (0, 1))
    block = psi_r.reshape(4, -1)
    out = gate4 @ block
    psi_r = out.reshape(2, 2, *psi_r.shape[2:])
    psi = np.moveaxis(psi_r, (0, 1), (a, b)).reshape(-1)
    return psi


def vqe_state(n: int, params: np.ndarray, L: int) -> np.ndarray:
    K = 1 << n
    psi = np.zeros(K, dtype=np.complex128)
    psi[0] = 1.0

    params = np.asarray(params, dtype=float)
    assert params.size == 2 * n * L

    for l in range(L):
        ry = params[l * (2 * n): l * (2 * n) + n]
        rz = params[l * (2 * n) + n: (l + 1) * (2 * n)]
        for q in range(n):
            cy, sy = math.cos(ry[q] / 2), math.sin(ry[q] / 2)
            RY = np.array([[cy, -sy], [sy, cy]], dtype=np.complex128)
            psi = _apply_1q(psi, RY, q, n)
            cz, sz = np.exp(-0.5j * rz[q]), np.exp(+0.5j * rz[q])
            RZ = np.array([[cz, 0], [0, sz]], dtype=np.complex128)
            psi = _apply_1q(psi, RZ, q, n)
        for q in range(n):
            psi = _apply_2q(psi, CNOT, q, (q + 1) % n, n)
        psi = _renorm_state(psi)
    return psi


def estimate_J_and_zexp(psi: np.ndarray, w: np.ndarray, ZZ: np.ndarray,
                        shots: int, rng: Optional[np.random.Generator]) -> Tuple[float, np.ndarray]:
    """
    Returns:
      J_hat, zexp_hat per edge (⟨Z_i Z_j⟩)
    """
    w = np.asarray(w, dtype=np.float64)
    ZZ = np.asarray(ZZ, dtype=np.int8)
    m, K = ZZ.shape

    probs = (psi.conj() * psi).real.astype(np.float64)
    s = float(np.sum(probs))
    if (not np.isfinite(s)) or s <= 0:
        probs[:] = 1.0 / probs.size
    else:
        probs /= s
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

    if shots is None or shots <= 0:
        # exact expectations
        zexp = (ZZ.astype(np.float64) @ probs)  # (m,)
    else:
        assert rng is not None
        idx = rng.choice(np.arange(K), size=int(shots), replace=True, p=probs)
        zexp = np.mean(ZZ[:, idx].astype(np.float64), axis=1)

    zexp = np.clip(np.nan_to_num(zexp, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
    p_cut = 0.5 * (1.0 - zexp)
    J = float(p_cut @ w)
    if not np.isfinite(J):
        J = 0.0
    return J, zexp


def vqe_eval(n: int, params: np.ndarray, L: int, w: np.ndarray, ZZ: np.ndarray,
             shots: int, rng_meas: Optional[np.random.Generator]) -> Tuple[float, np.ndarray]:
    psi = vqe_state(n, params, L)
    return estimate_J_and_zexp(psi, w, ZZ, shots, rng_meas)


# =============================================================================
# 6) Inner optimizer (SPSA)
# =============================================================================

def spsa_minimize(energy_fun, p0: np.ndarray, bounds: List[Tuple[float, float]],
                  iters: int, seed: int,
                  a: float = 0.2, c: float = 0.12, A: float = 20.0,
                  alpha: float = 0.602, gamma: float = 0.101) -> Tuple[np.ndarray, float, int]:
    rng = np.random.default_rng(to_uint_seed(seed))
    p = np.asarray(p0, dtype=np.float64).copy()
    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)

    best_p = p.copy()
    best_E = float("inf")
    evals = 0

    for k in range(1, iters + 1):
        ak = a / ((k + A) ** alpha)
        ck = c / (k ** gamma)
        delta = rng.choice([-1.0, 1.0], size=p.size)

        pp = np.clip(p + ck * delta, lo, hi)
        pm = np.clip(p - ck * delta, lo, hi)
        Ep = float(energy_fun(pp))
        Em = float(energy_fun(pm))
        evals += 2

        ghat = (Ep - Em) / (2.0 * ck) * delta
        p = np.clip(p - ak * ghat, lo, hi)

        E = float(energy_fun(p))
        evals += 1

        if E < best_E:
            best_E, best_p = E, p.copy()

    return best_p, best_E, evals


# =============================================================================
# 7) Classical diagnostic envelope (for normalization only)
# =============================================================================

def classical_Jstar_max(fam: Family1D, cut_mask: np.ndarray, grid_points: int) -> float:
    """
    Returns max_{λ in grid} max_z J(z;λ).
    Used only as a normalization scale for cross-instance comparability.
    """
    lams = np.linspace(fam.lam_min, fam.lam_max, int(grid_points))
    Jmax = -1e30
    for lam in lams:
        w = fam.w(float(lam))
        vals = cut_mask @ w  # (K,)
        v = float(np.max(vals))
        if v > Jmax:
            Jmax = v
    return float(Jmax)


# =============================================================================
# 8) Outer loops: ID vs BB-FD on F(λ)
# =============================================================================

@dataclass
class RunHist:
    evals: np.ndarray          # cumulative energy evaluations
    best: np.ndarray           # best-so-far value (unnormalized)
    best_norm: np.ndarray      # best-so-far normalized by J_star_max
    final_best_at_budget: float
    auc_at_budget: float


def run_outer(
    *,
    mode: str,
    n: int,
    edges: List[Tuple[int, int]],
    ZZ: np.ndarray,
    fam: Family1D,
    lam0: float,
    outer: int,
    inner: int,
    L: int,
    shots: int,
    seed: int,
    eta0: float,
    eta_pow: float,
    step_clip: Optional[float],
    c_frac: float,
    J_star_max: float,
    budget: float,
) -> RunHist:

    lam_min, lam_max = fam.lam_min, fam.lam_max
    lam = float(np.clip(lam0, lam_min, lam_max))

    D = 2 * n * L
    params = np.zeros(D, dtype=np.float64)
    bounds = [(-math.pi, math.pi)] * D

    # separate RNG for measurement noise (reproducible per run)
    rng_meas = np.random.default_rng(to_uint_seed(seed + 99991))

    evals_cum = 0.0
    best = -1e18

    evals_trace: List[float] = []
    best_trace: List[float] = []

    c = float(c_frac * (lam_max - lam_min))

    for t in range(1, outer + 1):

        # -------- inner solve at current λ (one value query F(λ)) ----------
        w = fam.w(lam)

        def energy_fun(pvec):
            J_hat, _ = vqe_eval(n, pvec, L, w, ZZ, shots, rng_meas)
            return -J_hat

        params, _, ev_in = spsa_minimize(
            energy_fun, params, bounds, iters=inner, seed=seed + 1000 * t
        )
        evals_cum += ev_in

        # Evaluate at the chosen params to get correlators (and J)
        J, zexp = vqe_eval(n, params, L, w, ZZ, shots, rng_meas)
        evals_cum += 1.0
        best = max(best, float(J))

        # -------- outer signal ----------
        if mode == "ID":
            p_cut = 0.5 * (1.0 - zexp)
            g = float(fam.dw_dlam(lam) @ p_cut)

        elif mode == "FD_VALUE":
            # Black-box FD on F(λ): needs extra inner solves at λ±c
            lp = float(np.clip(lam + c, lam_min, lam_max))
            lm = float(np.clip(lam - c, lam_min, lam_max))

            # +c
            w_p = fam.w(lp)

            def energy_fun_p(pvec):
                J_hat, _ = vqe_eval(n, pvec, L, w_p, ZZ, shots, rng_meas)
                return -J_hat

            p_p, _, evp = spsa_minimize(energy_fun_p, params, bounds, iters=inner, seed=seed + 1000 * t + 17)
            evals_cum += evp
            Jp, _ = vqe_eval(n, p_p, L, w_p, ZZ, shots, rng_meas)
            evals_cum += 1.0

            # -c
            w_m = fam.w(lm)

            def energy_fun_m(pvec):
                J_hat, _ = vqe_eval(n, pvec, L, w_m, ZZ, shots, rng_meas)
                return -J_hat

            p_m, _, evm = spsa_minimize(energy_fun_m, params, bounds, iters=inner, seed=seed + 1000 * t + 29)
            evals_cum += evm
            Jm, _ = vqe_eval(n, p_m, L, w_m, ZZ, shots, rng_meas)
            evals_cum += 1.0

            # FD has evaluated these candidates too -> count for best-so-far
            best = max(best, float(Jp), float(Jm))

            g = float((Jp - Jm) / (2.0 * c)) if c > 0 else 0.0

            # Optional warm-start: keep params from the closer perturbation to the next λ
            # (stronger baseline; uses already-paid computations)
            # We'll choose later once λ_{t+1} is known.

        else:
            raise ValueError("mode must be 'ID' or 'FD_VALUE'")

        # -------- matched outer update rule ----------
        eta_t = eta0 / (t ** eta_pow)
        step = float(eta_t * g)
        if step_clip is not None:
            step = float(np.clip(step, -step_clip, step_clip))
        lam_new = float(np.clip(lam + step, lam_min, lam_max))

        # For FD_VALUE: optional warm-start swap
        if mode == "FD_VALUE":
            # choose the params whose λ is closest to the new λ
            # (center params = params, plus/minus params = p_p / p_m)
            # If we didn't compute p_p/p_m due to some future refactor, skip safely.
            try:
                cand = [(lam, params), (lp, p_p), (lm, p_m)]
                lam_closest, p_closest = min(cand, key=lambda pr: abs(pr[0] - lam_new))
                params = np.asarray(p_closest, dtype=np.float64).copy()
            except Exception:
                pass

        lam = lam_new

        evals_trace.append(float(evals_cum))
        best_trace.append(float(best))

    evals_arr = np.asarray(evals_trace, dtype=float)
    best_arr = np.asarray(best_trace, dtype=float)
    best_norm = best_arr / max(1e-12, float(J_star_max))

    # metrics at a fixed budget
    B = float(budget)
    budgets = np.linspace(0.0, B, 200)
    best_on_grid = step_interp(evals_arr, best_norm, budgets)
    final_best = float(best_on_grid[-1])
    auc = auc_trapz(budgets, best_on_grid) / max(1e-12, B)  # normalized AUC

    return RunHist(
        evals=evals_arr,
        best=best_arr,
        best_norm=best_norm,
        final_best_at_budget=final_best,
        auc_at_budget=float(auc),
    )


# =============================================================================
# 9) Experiment runner + aggregation + table/plots
# =============================================================================

@dataclass
class AggCurves:
    budgets: np.ndarray
    mean_id: np.ndarray
    se_id: np.ndarray
    mean_fd: np.ndarray
    se_fd: np.ndarray
    final_id_mean: float
    final_id_se: float
    final_fd_mean: float
    final_fd_se: float
    auc_id_mean: float
    auc_id_se: float
    auc_fd_mean: float
    auc_fd_se: float
    budget_used: float


def aggregate_runs(runs_id: List[RunHist], runs_fd: List[RunHist], budget: float, budget_points: int) -> AggCurves:
    B = float(budget)
    budgets = np.linspace(0.0, B, int(budget_points))

    Y_id = np.stack([step_interp(r.evals, r.best_norm, budgets) for r in runs_id], axis=0)
    Y_fd = np.stack([step_interp(r.evals, r.best_norm, budgets) for r in runs_fd], axis=0)

    mean_id = np.mean(Y_id, axis=0)
    mean_fd = np.mean(Y_fd, axis=0)

    se_id = np.std(Y_id, axis=0, ddof=1) / math.sqrt(max(1, Y_id.shape[0])) if Y_id.shape[0] > 1 else np.full_like(mean_id, np.nan)
    se_fd = np.std(Y_fd, axis=0, ddof=1) / math.sqrt(max(1, Y_fd.shape[0])) if Y_fd.shape[0] > 1 else np.full_like(mean_fd, np.nan)

    final_id = np.array([r.final_best_at_budget for r in runs_id], dtype=float)
    final_fd = np.array([r.final_best_at_budget for r in runs_fd], dtype=float)
    auc_id = np.array([r.auc_at_budget for r in runs_id], dtype=float)
    auc_fd = np.array([r.auc_at_budget for r in runs_fd], dtype=float)

    return AggCurves(
        budgets=budgets,
        mean_id=mean_id,
        se_id=se_id,
        mean_fd=mean_fd,
        se_fd=se_fd,
        final_id_mean=float(np.mean(final_id)),
        final_id_se=stderr(final_id),
        final_fd_mean=float(np.mean(final_fd)),
        final_fd_se=stderr(final_fd),
        auc_id_mean=float(np.mean(auc_id)),
        auc_id_se=stderr(auc_id),
        auc_fd_mean=float(np.mean(auc_fd)),
        auc_fd_se=stderr(auc_fd),
        budget_used=B,
    )


def plot_budget_curves(path: Path, curves_by_shots: Dict[int, AggCurves], shots_list: List[int]):
    """
    If len(shots_list)==2 -> 1x2 panels (exact vs finite shots).
    Otherwise -> one panel per figure.
    """
    set_pub_style(grid=False, base_size=8)

    if len(shots_list) == 2:
        fig, axs = plt.subplots(1, 2, figsize=(FULL_W, H), constrained_layout=True, sharey=True)
        axes = list(axs)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(COL_W, H), constrained_layout=True)
        axes = [ax]

    for ax, shots in zip(axes, shots_list):
        C = curves_by_shots[shots]
        x = C.budgets

        ax.plot(x, C.mean_id, color=COLORS["ID"], lw=1.8, label="VQE + ID")
        ax.fill_between(x, C.mean_id - C.se_id, C.mean_id + C.se_id, color=COLORS["ID"], alpha=0.18, linewidth=0)

        ax.plot(x, C.mean_fd, color=COLORS["FD"], lw=1.8, ls="--", label=r"VQE + BB-FD (re-solve)")
        ax.fill_between(x, C.mean_fd - C.se_fd, C.mean_fd + C.se_fd, color=COLORS["FD"], alpha=0.18, linewidth=0)

        # ground truth line (diagnostic; 1.0 means reaching classical envelope max)
        ax.axhline(1.0, color=COLORS["GT"], lw=1.0, ls=":", alpha=0.9)

        ax.set_xlim(0, C.budget_used)
        ax.set_ylim(0.0, 1.05)

        ax.set_xlabel("Energy evaluations")

        # small panel annotation (not a title)
        txt = "exact" if shots <= 0 else f"S={shots}"
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, ha="left", va="top")

        if ax is axes[0]:
            ax.set_ylabel(r"Best-so-far $\hat F / J_{\mathrm{cl}}^*$")

        ax.legend(loc="lower right", frameon=False)

    _savefig(fig, path)
    plt.close(fig)


def write_table_csv(path: Path, rows: List[Dict[str, object]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_table_latex(path: Path, table: Dict[Tuple[str, str], Dict[str, Tuple[float, float]]]):
    """
    table[(kind, shots_label)]["ID"] = (mean, se), same for "FD".
    """
    lines = []
    lines.append(r"\begin{tabular}{l c c c}")
    lines.append(r"\hline")
    lines.append(r"Family & Shots & VQE--ID & VQE--BB-FD (re-solve) \\")
    lines.append(r"\hline")
    for (kind, shots_lbl), d in table.items():
        mid, seid = d["ID"]
        mfd, sefd = d["FD"]
        lines.append(f"{kind} & {shots_lbl} & {mid:.3f} $\\pm$ {seid:.3f} & {mfd:.3f} $\\pm$ {sefd:.3f} \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_str_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="out_exp0A")
    p.add_argument("--fmt", type=str, default="pdf", choices=["pdf", "png", "svg"])

    # Instances / repetitions
    p.add_argument("--seed0", type=int, default=7, help="Base seed; instance seeds are seed0+i.")
    p.add_argument("--num_seeds", type=int, default=10)

    # Problem size
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--p_edge", type=float, default=0.45)
    p.add_argument("--lam_min", type=float, default=-5.0)
    p.add_argument("--lam_max", type=float, default=5.0)
    p.add_argument("--lam0", type=float, default=0.8)

    # Canonical family parameters
    p.add_argument("--families", type=str, default="linear,quadratic,periodic")
    p.add_argument("--periodic_K", type=int, default=6)

    # Shot settings: comma-separated; use 0 for exact
    p.add_argument("--shots_list", type=str, default="0,256")

    # Inner/outer
    p.add_argument("--outer", type=int, default=30)
    p.add_argument("--inner", type=int, default=28)
    p.add_argument("--L", type=int, default=2)
    p.add_argument("--eta0", type=float, default=0.35)
    p.add_argument("--eta_pow", type=float, default=0.6)
    p.add_argument("--step_clip", type=float, default=0.6)
    p.add_argument("--c_frac", type=float, default=0.05)

    # Classical normalization grid
    p.add_argument("--grid", type=int, default=401)

    # Budget axis (table + plots)
    p.add_argument("--budget", type=float, default=2500.0)
    p.add_argument("--budget_points", type=int, default=240)

    return p.parse_args()


def main():
    a = parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    families = parse_str_list(a.families)
    shots_list = parse_int_list(a.shots_list)
    if len(shots_list) == 0:
        raise ValueError("--shots_list must contain at least one integer (0 for exact).")

    # Precompute Z once (depends only on n)
    Z = precompute_z_big_endian(a.n)

    # Storage for table
    table_rows: List[Dict[str, object]] = []
    latex_table: Dict[Tuple[str, str], Dict[str, Tuple[float, float]]] = {}

    # Per-run metrics CSV
    per_run_rows: List[Dict[str, object]] = []

    for kind in families:
        curves_by_shots: Dict[int, AggCurves] = {}

        for shots in shots_list:
            runs_id: List[RunHist] = []
            runs_fd: List[RunHist] = []

            for i in range(a.num_seeds):
                seed = a.seed0 + i
                rng_inst = np.random.default_rng(to_uint_seed(seed))

                edges = generate_random_graph(a.n, a.p_edge, rng_inst)
                if not edges:
                    continue

                ZZ, cut_mask = precompute_ZZ_and_cutmask(edges, Z)

                fam = Family1D(
                    m=len(edges),
                    kind=kind,
                    lam_bounds=(a.lam_min, a.lam_max),
                    rng=rng_inst,
                    periodic_K=a.periodic_K,
                )

                J_star_max = classical_Jstar_max(fam, cut_mask, a.grid)

                budget = float(a.budget)

                hist_id = run_outer(
                    mode="ID",
                    n=a.n,
                    edges=edges,
                    ZZ=ZZ,
                    fam=fam,
                    lam0=a.lam0,
                    outer=a.outer,
                    inner=a.inner,
                    L=a.L,
                    shots=shots,
                    seed=to_uint_seed(seed + 111),
                    eta0=a.eta0,
                    eta_pow=a.eta_pow,
                    step_clip=a.step_clip,
                    c_frac=a.c_frac,
                    J_star_max=J_star_max,
                    budget=budget,
                )
                hist_fd = run_outer(
                    mode="FD_VALUE",
                    n=a.n,
                    edges=edges,
                    ZZ=ZZ,
                    fam=fam,
                    lam0=a.lam0,
                    outer=a.outer,
                    inner=a.inner,
                    L=a.L,
                    shots=shots,
                    seed=to_uint_seed(seed + 222),
                    eta0=a.eta0,
                    eta_pow=a.eta_pow,
                    step_clip=a.step_clip,
                    c_frac=a.c_frac,
                    J_star_max=J_star_max,
                    budget=budget,
                )

                runs_id.append(hist_id)
                runs_fd.append(hist_fd)

                per_run_rows.append({
                    "kind": kind,
                    "shots": shots,
                    "seed": seed,
                    "m_edges": len(edges),
                    "J_star_max": J_star_max,
                    "ID_final@B": hist_id.final_best_at_budget,
                    "FD_final@B": hist_fd.final_best_at_budget,
                    "ID_auc@B": hist_id.auc_at_budget,
                    "FD_auc@B": hist_fd.auc_at_budget,
                    "ID_evals_end": float(hist_id.evals[-1]) if hist_id.evals.size else float("nan"),
                    "FD_evals_end": float(hist_fd.evals[-1]) if hist_fd.evals.size else float("nan"),
                })

            if len(runs_id) < 2:
                print(f"[warn] kind={kind}, shots={shots}: only {len(runs_id)} runs (need >=2 for stderr).")

            # Make sure budget is covered by all runs (avoid extrapolating flat tails)
            min_cover = min(float(np.max(r.evals)) for r in (runs_id + runs_fd) if r.evals.size)
            budget_used = min(float(a.budget), min_cover)

            curves = aggregate_runs(runs_id, runs_fd, budget=budget_used, budget_points=a.budget_points)
            curves_by_shots[shots] = curves

            shots_lbl = "exact" if shots <= 0 else str(shots)

            table_rows.append({
                "family": kind,
                "shots": shots_lbl,
                "budget": f"{budget_used:.0f}",
                "ID_mean": f"{curves.final_id_mean:.4f}",
                "ID_stderr": f"{curves.final_id_se:.4f}",
                "FD_mean": f"{curves.final_fd_mean:.4f}",
                "FD_stderr": f"{curves.final_fd_se:.4f}",
                "ID_auc_mean": f"{curves.auc_id_mean:.4f}",
                "ID_auc_stderr": f"{curves.auc_id_se:.4f}",
                "FD_auc_mean": f"{curves.auc_fd_mean:.4f}",
                "FD_auc_stderr": f"{curves.auc_fd_se:.4f}",
            })

            latex_table[(kind, shots_lbl)] = {
                "ID": (curves.final_id_mean, curves.final_id_se),
                "FD": (curves.final_fd_mean, curves.final_fd_se),
            }

            print(f"[done] kind={kind:9s} shots={shots_lbl:>5s} | "
                  f"ID {curves.final_id_mean:.3f}±{curves.final_id_se:.3f} | "
                  f"FD {curves.final_fd_mean:.3f}±{curves.final_fd_se:.3f} | B={budget_used:.0f}")

        # Plot for this family
        if len(shots_list) == 2:
            _save = out / f"fig0A_budget_{kind}.{a.fmt}"
            plot_budget_curves(_save, curves_by_shots, shots_list=shots_list)
        else:
            for shots in shots_list:
                _save = out / f"fig0A_budget_{kind}_shots{shots}.{a.fmt}"
                plot_budget_curves(_save, {shots: curves_by_shots[shots]}, shots_list=[shots])

    write_table_csv(out / "table0A_summary.csv", table_rows)
    write_table_csv(out / "runs0A_metrics.csv", per_run_rows)
    write_table_latex(out / "table0A_summary.tex", latex_table)

    print("\nSaved to:", out.resolve())
    print(" - fig0A_budget_<family>.<fmt>")
    print(" - table0A_summary.csv / table0A_summary.tex")
    print(" - runs0A_metrics.csv")


if __name__ == "__main__":
    main()
