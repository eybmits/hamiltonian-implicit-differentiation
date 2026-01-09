#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp0A_budget_efficiency.py
=========================

Experiment 0A (NPJ Core): Budget Efficiency of ID vs Black-box FD on the *value function* F(λ)
----------------------------------------------------------------------------------------------
[FINAL VERSION: t=20 MARKER VISUALIZATION]

We study a 1D parametrized Max-Cut Hamiltonian family

    H(λ) = Σ_e w_e(λ) C_e ,     C_e = (I - Z_i Z_j)/2 ,

and the VQE value function (bilevel target)

    F(λ) = max_θ  J(θ,λ) ,      J(θ,λ) = ⟨H(λ)⟩_θ = Σ_e w_e(λ) p_cut,e(θ) ,
                               p_cut,e = (1 - ⟨Z_i Z_j⟩)/2 .

Outer-loop signals compared under a *matched outer update rule* (same step-size schedule):
  - ID: Cheap gradient (1 inner solve).
  - FD: Expensive gradient (2-3 inner solves).

VISUALIZATION FEATURE:
  The plot includes a specific marker at outer iteration t=20.
  Since FD is more expensive per step, its t=20 marker appears much further
  to the right (higher budget) than ID's t=20 marker. This visualizes the
  efficiency gap immediately: "Both are at step 20, but ID is much cheaper."
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
# Silence noisy-but-harmless PDF font timestamp chatter
# -----------------------------------------------------------------------------
warnings.filterwarnings("ignore", message=".*timestamp seems very low.*")
warnings.filterwarnings("ignore", message=".*regarding as unix timestamp.*")
_ft = logging.getLogger("fontTools")
_ft.setLevel(logging.ERROR)
_ft.propagate = False
if not _ft.handlers:
    _ft.addHandler(logging.NullHandler())


# =============================================================================
# 1) Pub-ish plotting (CLEAN STYLE)
# =============================================================================

COLORS = {
    "ID": "#D62728",   # Red
    "FD": "#1F77B4",   # Blue
    "GT": "#000000",   # Black (Ground Truth)
}

def set_pub_style(grid: bool = True, base_size: int = 9):
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
        "xtick.top": False,
        "ytick.right": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.grid": grid,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "savefig.transparent": False,
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
    evals = np.asarray(evals, dtype=float)
    values = np.asarray(values, dtype=float)
    budgets = np.asarray(budgets, dtype=float)
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
    K = 1 << n
    idx = np.arange(K, dtype=np.uint32)
    Z = np.empty((n, K), dtype=np.int8)
    for q in range(n):
        bitpos = n - 1 - q
        Z[q] = 1 - 2 * ((idx >> bitpos) & 1).astype(np.int8)
    return Z


def precompute_ZZ_and_cutmask(edges: List[Tuple[int, int]], Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m = len(edges)
    K = Z.shape[1]
    ZZ = np.empty((m, K), dtype=np.int8)
    for e, (i, j) in enumerate(edges):
        ZZ[e] = (Z[i] * Z[j]).astype(np.int8)
    cut_mask = 0.5 * (1.0 - ZZ.astype(np.float64)).T
    return ZZ, cut_mask


# =============================================================================
# 4) Canonical 1D families
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

        f_min = {
            "linear": -math.sqrt(3.0),
            "quadratic": -math.sqrt(45.0/4.0) * (1.0/3.0),
            "periodic": -math.sqrt(2.0),
        }[kind]
        w_min_target = 0.05
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
# 5) VQE ansatz
# =============================================================================

CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex128)

def _renorm_state(psi: np.ndarray) -> np.ndarray:
    nrm = float(np.vdot(psi, psi).real)
    if (not np.isfinite(nrm)) or nrm <= 0:
        psi[:] = 1.0 / math.sqrt(psi.size)
    else:
        psi /= math.sqrt(nrm)
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
    if q1 == q2: return psi
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
    w = np.asarray(w, dtype=np.float64)
    ZZ = np.asarray(ZZ, dtype=np.int8)
    m, K = ZZ.shape
    probs = (psi.conj() * psi).real.astype(np.float64)
    s = float(np.sum(probs))
    if (not np.isfinite(s)) or s <= 0: probs[:] = 1.0 / probs.size
    else: probs /= s
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

    if shots is None or shots <= 0:
        zexp = (ZZ.astype(np.float64) @ probs)
    else:
        idx = rng.choice(np.arange(K), size=int(shots), replace=True, p=probs)
        zexp = np.mean(ZZ[:, idx].astype(np.float64), axis=1)

    zexp = np.clip(np.nan_to_num(zexp, nan=0.0), -1.0, 1.0)
    p_cut = 0.5 * (1.0 - zexp)
    J = float(p_cut @ w)
    if not np.isfinite(J): J = 0.0
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
        if E < best_E: best_E, best_p = E, p.copy()

    return best_p, best_E, evals


# =============================================================================
# 7) Classical diagnostic envelope
# =============================================================================

def classical_Jstar_max(fam: Family1D, cut_mask: np.ndarray, grid_points: int) -> float:
    lams = np.linspace(fam.lam_min, fam.lam_max, int(grid_points))
    Jmax = -1e30
    for lam in lams:
        w = fam.w(float(lam))
        vals = cut_mask @ w
        v = float(np.max(vals))
        if v > Jmax: Jmax = v
    return float(Jmax)


# =============================================================================
# 8) Outer loops
# =============================================================================

@dataclass
class RunHist:
    evals: np.ndarray
    best: np.ndarray
    best_norm: np.ndarray
    final_best_at_budget: float
    auc_at_budget: float


def run_outer(
    *, mode: str, n: int, edges: List[Tuple[int, int]], ZZ: np.ndarray,
    fam: Family1D, lam0: float, outer: int, inner: int, L: int, shots: int,
    seed: int, eta0: float, eta_pow: float, step_clip: Optional[float],
    c_frac: float, J_star_max: float, budget: float,
) -> RunHist:

    lam_min, lam_max = fam.lam_min, fam.lam_max
    lam = float(np.clip(lam0, lam_min, lam_max))
    D = 2 * n * L
    params = np.zeros(D, dtype=np.float64)
    bounds = [(-math.pi, math.pi)] * D
    rng_meas = np.random.default_rng(to_uint_seed(seed + 99991))

    evals_cum = 0.0
    best = -1e18
    evals_trace: List[float] = []
    best_trace: List[float] = []
    c = float(c_frac * (lam_max - lam_min))

    for t in range(1, outer + 1):
        w = fam.w(lam)
        def energy_fun(pvec):
            J_hat, _ = vqe_eval(n, pvec, L, w, ZZ, shots, rng_meas)
            return -J_hat

        params, _, ev_in = spsa_minimize(energy_fun, params, bounds, iters=inner, seed=seed + 1000 * t)
        evals_cum += ev_in

        J, zexp = vqe_eval(n, params, L, w, ZZ, shots, rng_meas)
        evals_cum += 1.0
        best = max(best, float(J))

        if mode == "ID":
            p_cut = 0.5 * (1.0 - zexp)
            g = float(fam.dw_dlam(lam) @ p_cut)

        elif mode == "FD_VALUE":
            lp = float(np.clip(lam + c, lam_min, lam_max))
            lm = float(np.clip(lam - c, lam_min, lam_max))
            w_p = fam.w(lp)
            def energy_fun_p(pvec):
                J_hat, _ = vqe_eval(n, pvec, L, w_p, ZZ, shots, rng_meas)
                return -J_hat
            p_p, _, evp = spsa_minimize(energy_fun_p, params, bounds, iters=inner, seed=seed + 1000 * t + 17)
            evals_cum += evp
            Jp, _ = vqe_eval(n, p_p, L, w_p, ZZ, shots, rng_meas)
            evals_cum += 1.0

            w_m = fam.w(lm)
            def energy_fun_m(pvec):
                J_hat, _ = vqe_eval(n, pvec, L, w_m, ZZ, shots, rng_meas)
                return -J_hat
            p_m, _, evm = spsa_minimize(energy_fun_m, params, bounds, iters=inner, seed=seed + 1000 * t + 29)
            evals_cum += evm
            Jm, _ = vqe_eval(n, p_m, L, w_m, ZZ, shots, rng_meas)
            evals_cum += 1.0

            best = max(best, float(Jp), float(Jm))
            g = float((Jp - Jm) / (2.0 * c)) if c > 0 else 0.0

            try:
                cand = [(lam, params), (lp, p_p), (lm, p_m)]
                lam_closest, p_closest = min(cand, key=lambda pr: abs(pr[0] - lam_new))
                params = np.asarray(p_closest, dtype=np.float64).copy()
            except Exception: pass
        else:
            raise ValueError("mode?")

        eta_t = eta0 / (t ** eta_pow)
        step = float(eta_t * g)
        if step_clip is not None: step = float(np.clip(step, -step_clip, step_clip))
        lam_new = float(np.clip(lam + step, lam_min, lam_max))
        lam = lam_new

        evals_trace.append(float(evals_cum))
        best_trace.append(float(best))

    evals_arr = np.asarray(evals_trace, dtype=float)
    best_arr = np.asarray(best_trace, dtype=float)
    best_norm = best_arr / max(1e-12, float(J_star_max))

    B = float(budget)
    budgets = np.linspace(0.0, B, 200)
    best_on_grid = step_interp(evals_arr, best_norm, budgets)
    final_best = float(best_on_grid[-1])
    auc = auc_trapz(budgets, best_on_grid) / max(1e-12, B)

    return RunHist(evals=evals_arr, best=best_arr, best_norm=best_norm,
                   final_best_at_budget=final_best, auc_at_budget=float(auc))


# =============================================================================
# 9) Aggregation
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
    # Comparison coords for t=20
    id_p20: Optional[Tuple[float, float]]
    fd_p20: Optional[Tuple[float, float]]


def aggregate_runs(runs_id: List[RunHist], runs_fd: List[RunHist], budget: float, budget_points: int) -> AggCurves:
    B = float(budget)
    budgets = np.linspace(0.0, B, int(budget_points))

    Y_id = np.stack([step_interp(r.evals, r.best_norm, budgets) for r in runs_id], axis=0)
    Y_fd = np.stack([step_interp(r.evals, r.best_norm, budgets) for r in runs_fd], axis=0)

    mean_id = np.mean(Y_id, axis=0)
    mean_fd = np.mean(Y_fd, axis=0)
    
    se_id = np.std(Y_id, axis=0, ddof=1) / math.sqrt(max(1, Y_id.shape[0]))
    se_fd = np.std(Y_fd, axis=0, ddof=1) / math.sqrt(max(1, Y_fd.shape[0]))

    final_id = np.array([r.final_best_at_budget for r in runs_id], dtype=float)
    final_fd = np.array([r.final_best_at_budget for r in runs_fd], dtype=float)
    auc_id = np.array([r.auc_at_budget for r in runs_id], dtype=float)
    auc_fd = np.array([r.auc_at_budget for r in runs_fd], dtype=float)

    # --- Find coord for Outer Iteration 20 (Index 19) ---
    target_idx = 19
    def get_avg_coord(runs, idx):
        xs, ys = [], []
        for r in runs:
            # Check if this run actually reached the target step
            if idx < r.evals.size:
                xs.append(r.evals[idx])
                ys.append(r.best_norm[idx])
        if not xs: return None
        return (float(np.mean(xs)), float(np.mean(ys)))

    id_p20 = get_avg_coord(runs_id, target_idx)
    fd_p20 = get_avg_coord(runs_fd, target_idx)

    return AggCurves(
        budgets=budgets, mean_id=mean_id, se_id=se_id,
        mean_fd=mean_fd, se_fd=se_fd,
        final_id_mean=float(np.mean(final_id)), final_id_se=stderr(final_id),
        final_fd_mean=float(np.mean(final_fd)), final_fd_se=stderr(final_fd),
        auc_id_mean=float(np.mean(auc_id)), auc_id_se=stderr(auc_id),
        auc_fd_mean=float(np.mean(auc_fd)), auc_fd_se=stderr(auc_fd),
        budget_used=B,
        id_p20=id_p20, fd_p20=fd_p20
    )


def plot_budget_curves(path: Path, curves_by_shots: Dict[int, AggCurves], shots_list: List[int]):
    set_pub_style(grid=True, base_size=9)
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['grid.alpha'] = 0.25
    mpl.rcParams['grid.linestyle'] = '--'

    if len(shots_list) == 2:
        fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True, sharey=True)
        axes = list(axs)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.0), constrained_layout=True)
        axes = [ax]

    for ax, shots in zip(axes, shots_list):
        C = curves_by_shots[shots]
        x = C.budgets

        # GT Line
        ax.axhline(1.0, color=COLORS["GT"], lw=1.0, ls=":", alpha=0.6, label="_nolegend_")

        # FD (Blue)
        ax.plot(x, C.mean_fd, color=COLORS["FD"], lw=1.5, ls="--", label="VQE + FD")
        ax.fill_between(x, C.mean_fd - C.se_fd, C.mean_fd + C.se_fd, color=COLORS["FD"], alpha=0.15, linewidth=0)

        # ID (Red)
        ax.plot(x, C.mean_id, color=COLORS["ID"], lw=1.8, ls="-", label="VQE + ID")
        ax.fill_between(x, C.mean_id - C.se_id, C.mean_id + C.se_id, color=COLORS["ID"], alpha=0.2, linewidth=0)

        # Limits
        ax.set_xlim(0, C.budget_used)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("Energy evaluations")
        if ax is axes[0]:
            ax.set_ylabel(r"Norm. Approx. Ratio $\hat{F} / J_{\mathrm{cl}}^*$")

        # --- MARKER LOGIC FOR t=20 ---
        # Marker for ID (t=20)
        if C.id_p20:
            x_id, y_id = C.id_p20
            # Circle marker for ID
            ax.plot(x_id, y_id, marker='o', color=COLORS["ID"], markersize=6, 
                    zorder=10, markeredgecolor='white', markeredgewidth=1.0)
            ax.text(x_id, y_id + 0.05, "t=20", color=COLORS["ID"], fontsize=8, 
                    ha="center", fontweight="bold")

        # Marker for FD (t=20)
        if C.fd_p20:
            x_fd, y_fd = C.fd_p20
            # Only draw if inside current budget limits
            if x_fd <= C.budget_used:
                # Square marker for FD
                ax.plot(x_fd, y_fd, marker='s', color=COLORS["FD"], markersize=6, 
                        zorder=10, markeredgecolor='white', markeredgewidth=1.0)
                ax.text(x_fd, y_fd - 0.08, "t=20", color=COLORS["FD"], fontsize=8, 
                        ha="center", fontweight="bold")
                
                # Optional: Connect them to emphasize they are the same step
                if C.id_p20:
                    ax.annotate("", xy=(x_id, y_id), xytext=(x_fd, y_fd),
                                arrowprops=dict(arrowstyle="-", color="gray",
                                                linestyle=":", linewidth=1.0, alpha=0.6))

        ax.legend(loc="lower right", frameon=False, fontsize=8)

    _savefig(fig, path)
    plt.close(fig)


def write_table_csv(path: Path, rows: List[Dict[str, object]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows: w.writerow(r)

def write_table_latex(path: Path, table: Dict[Tuple[str, str], Dict[str, Tuple[float, float]]]):
    lines = [r"\begin{tabular}{l c c c}", r"\hline", r"Family & Shots & VQE--ID & VQE--FD \\", r"\hline"]
    for (kind, shots_lbl), d in table.items():
        mid, seid = d["ID"]
        mfd, sefd = d["FD"]
        lines.append(f"{kind} & {shots_lbl} & {mid:.3f} $\\pm$ {seid:.3f} & {mfd:.3f} $\\pm$ {sefd:.3f} \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))

def parse_int_list(s: str) -> List[int]: return [int(x.strip()) for x in s.split(",") if x.strip()]
def parse_str_list(s: str) -> List[str]: return [x.strip() for x in s.split(",") if x.strip()]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="out_exp0A")
    p.add_argument("--fmt", type=str, default="pdf", choices=["pdf", "png", "svg"])
    p.add_argument("--seed0", type=int, default=1)
    p.add_argument("--num_seeds", type=int, default=3)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--p_edge", type=float, default=0.45)
    p.add_argument("--lam_min", type=float, default=-5.0)
    p.add_argument("--lam_max", type=float, default=5.0)
    p.add_argument("--lam0", type=float, default=0.8)
    p.add_argument("--families", type=str, default="linear,quadratic,periodic")
    p.add_argument("--periodic_K", type=int, default=6)
    p.add_argument("--shots_list", type=str, default="0,256")
    p.add_argument("--outer", type=int, default=100)
    p.add_argument("--inner", type=int, default=10)
    p.add_argument("--L", type=int, default=2)
    p.add_argument("--eta0", type=float, default=0.35)
    p.add_argument("--eta_pow", type=float, default=0.6)
    p.add_argument("--step_clip", type=float, default=0.6)
    p.add_argument("--c_frac", type=float, default=0.05)
    p.add_argument("--grid", type=int, default=401)
    p.add_argument("--budget", type=float, default=2000.0)
    p.add_argument("--budget_points", type=int, default=500)
    return p.parse_args()

def main():
    a = parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    families = parse_str_list(a.families)
    shots_list = parse_int_list(a.shots_list)
    Z = precompute_z_big_endian(a.n)
    table_rows = []
    latex_table = {}
    per_run_rows = []

    for kind in families:
        curves_by_shots = {}
        for shots in shots_list:
            runs_id, runs_fd = [], []
            for i in range(a.num_seeds):
                seed = a.seed0 + i
                rng_inst = np.random.default_rng(to_uint_seed(seed))
                edges = generate_random_graph(a.n, a.p_edge, rng_inst)
                if not edges: continue
                ZZ, cut_mask = precompute_ZZ_and_cutmask(edges, Z)
                fam = Family1D(len(edges), kind, (a.lam_min, a.lam_max), rng_inst, a.periodic_K)
                J_star_max = classical_Jstar_max(fam, cut_mask, a.grid)
                budget = float(a.budget)

                runs_id.append(run_outer(mode="ID", n=a.n, edges=edges, ZZ=ZZ, fam=fam, lam0=a.lam0,
                                         outer=a.outer, inner=a.inner, L=a.L, shots=shots,
                                         seed=to_uint_seed(seed + 111), eta0=a.eta0, eta_pow=a.eta_pow,
                                         step_clip=a.step_clip, c_frac=a.c_frac, J_star_max=J_star_max, budget=budget))
                runs_fd.append(run_outer(mode="FD_VALUE", n=a.n, edges=edges, ZZ=ZZ, fam=fam, lam0=a.lam0,
                                         outer=a.outer, inner=a.inner, L=a.L, shots=shots,
                                         seed=to_uint_seed(seed + 222), eta0=a.eta0, eta_pow=a.eta_pow,
                                         step_clip=a.step_clip, c_frac=a.c_frac, J_star_max=J_star_max, budget=budget))

            min_cover = min(float(np.max(r.evals)) for r in (runs_id + runs_fd) if r.evals.size)
            budget_used = min(float(a.budget), min_cover)
            curves = aggregate_runs(runs_id, runs_fd, budget=budget_used, budget_points=a.budget_points)
            curves_by_shots[shots] = curves
            shots_lbl = "exact" if shots <= 0 else str(shots)

            table_rows.append({"family": kind, "shots": shots_lbl, "budget": f"{budget_used:.0f}",
                               "ID_mean": f"{curves.final_id_mean:.4f}", "ID_stderr": f"{curves.final_id_se:.4f}",
                               "FD_mean": f"{curves.final_fd_mean:.4f}", "FD_stderr": f"{curves.final_fd_se:.4f}"})
            latex_table[(kind, shots_lbl)] = {"ID": (curves.final_id_mean, curves.final_id_se),
                                              "FD": (curves.final_fd_mean, curves.final_fd_se)}
            print(f"[done] {kind:9s} {shots_lbl:>5s} | ID {curves.final_id_mean:.3f} | FD {curves.final_fd_mean:.3f}")

        if len(shots_list) == 2:
            plot_budget_curves(out / f"fig0A_budget_{kind}.{a.fmt}", curves_by_shots, shots_list)
        else:
            for shots in shots_list:
                plot_budget_curves(out / f"fig0A_budget_{kind}_shots{shots}.{a.fmt}", {shots: curves_by_shots[shots]}, [shots])

    write_table_csv(out / "table0A_summary.csv", table_rows)
    write_table_latex(out / "table0A_summary.tex", latex_table)
    print("\nSaved to:", out.resolve())

if __name__ == "__main__":
    main()