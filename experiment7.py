#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp7_edgewise_lambda_vector.py
==============================

Experiment 7 (Main / Supplement): Edge-wise outer parameters (vector λ)
-----------------------------------------------------------------------

Goal
----
Generalize the outer control from a single scalar λ to a vector of per-edge controls

    λ = (λ_e)_{e∈E},   with   H(λ) = Σ_{e∈E} w_e(λ_e) P_e,  P_e=(I - Z_i Z_j)/2.

We test whether the correlator-reuse implicit differentiation (CR-ImpDiff / "ID")
advantage persists in this higher-dimensional outer problem under matched
evaluation budgets.

Key objects (aligned with the manuscript)
----------------------------------------
Inner objective:
  J(φ, λ_vec) = ⟨H(λ_vec)⟩_φ = Σ_e w_e(λ_e) p_e(φ),    p_e=(1-⟨Z_i Z_j⟩)/2 ∈ [0,1].

Bilevel value function (target):
  F(λ_vec) = max_φ J(φ, λ_vec).

Methods (outer direction only differs; same outer update rule)
--------------------------------------------------------------
(1) ID / CR-ImpDiff (envelope / partial derivative signal)
    After ONE inner solve at current λ_vec:
        g_ID,e = ∂J/∂λ_e = w'_e(λ_e) * p_e
    i.e. the full gradient vector is obtained by reusing the same ZZ correlators
    already needed for J. (Negligible additional quantum measurement cost.)

(2) Black-box bilevel baseline in high dimension: SPSA finite difference on F
    Use a Rademacher direction Δ ∈ {±1}^m:
        g_FD = [F(λ_vec + cΔ) - F(λ_vec - cΔ)]/(2c) * Δ
    Each F(·) query requires an additional inner re-solve at the perturbed λ.
    Thus FD costs ~3 inner solves per outer step (center + plus + minus).

Fairness / cost model
---------------------
- We compare at a fixed total evaluation budget B (energy evaluations).
- For FD, best-so-far is computed over ALL value queries it actually makes
  (center, plus, minus). We do not discard evaluated candidates.
- We report performance as best-so-far value vs cumulative energy evaluations,
  and as normalized AUC_B over [0,B].

Normalization J*
----------------
In the edge-wise setting with positive weights, an informative diagnostic scale is:
  w_e^max = max_{λ∈[λmin,λmax]} w_e(λ)   (independent per edge),
  J* = max_{z∈{0,1}^n} Σ_e w_e^max * I{edge e cut by z}.
We compute w^max by scanning a λ-grid and compute J* by enumerating all 2^n bitstrings
(feasible for n<=12). J* is used only for normalization/diagnostics.

Outputs (paper-ready)
---------------------
Saved in --out directory:
  - fig7_edgewise_budget_gain_<suffix>.<fmt>
      2-panel figure: (a) best-so-far / J* vs evaluations (mean±stderr),
                      (b) per-instance ΔAUC_B vs |E| (scatter + summary text).
  - fig7_edgewise_steps_<suffix>.<fmt>
      Bar chart: outer steps completed within budget B (mean±stderr).
  - runs7_edgewise_metrics.csv
      Per-instance metrics (AUC, final best at budget, steps, costs, etc.).
  - table7_edgewise_summary.csv / .tex
      Mean±stderr summary across instances.
  - exp7_edgewise_summary.txt
      Compact copy/paste summary.

Example
-------
python exp7_edgewise_lambda_vector.py \
  --family periodic --periodic_K 6 \
  --n 12 --p_edge 0.45 \
  --inner_iters 28 --restarts 1 --L 2 \
  --budget_evals 5100 \
  --num_instances 20 \
  --shots 0 \
  --fmt pdf --out out_exp7

"""

import math
import argparse
import csv
import warnings
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# ------------------------------------------------------------------------------
# Silence noisy-but-harmless fontTools PDF logs and a few numpy warnings
# ------------------------------------------------------------------------------
warnings.filterwarnings("ignore", message=".*timestamp seems very low.*")
warnings.filterwarnings("ignore", message=".*regarding as unix timestamp.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*encountered in matmul.*")

_ft = logging.getLogger("fontTools")
_ft.setLevel(logging.ERROR)
_ft.propagate = False
if not _ft.handlers:
    _ft.addHandler(logging.NullHandler())
logging.getLogger("fontTools.ttLib").setLevel(logging.ERROR)
logging.getLogger("fontTools.subset").setLevel(logging.ERROR)


# ==============================================================================
# 1) Paper-ish plotting style (NPJ/Nature-ish)
# ==============================================================================

COLORS = {
    "ID":  "#D62728",  # red
    "FD":  "#1F77B4",  # blue
    "GT":  "#000000",
}

FULL_W = 6.95
H_TWO = 2.6
COL_W = 3.37


def set_pub_style(grid: bool = False):
    mpl.rcdefaults()
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Liberation Serif"],
        "font.size": 8,
        "axes.labelsize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
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


def _mean_stderr(x: np.ndarray, axis: int = 0):
    x = np.asarray(x, float)
    mu = np.nanmean(x, axis=axis)
    sd = np.nanstd(x, axis=axis, ddof=1)
    n = np.sum(np.isfinite(x), axis=axis)
    se = sd / np.sqrt(np.maximum(1, n))
    return mu, se


def _step_interp(evals: np.ndarray, vals: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """
    Piecewise-constant interpolation (step function):
      y(b) = last observed value with eval <= b.
    """
    evals = np.asarray(evals, float)
    vals = np.asarray(vals, float)
    grid = np.asarray(grid, float)

    if evals.size == 0:
        return np.full_like(grid, np.nan, dtype=float)

    order = np.argsort(evals)
    evals = evals[order]
    vals = vals[order]

    # Anchor at budget 0
    if evals[0] > 0.0:
        evals = np.concatenate([[0.0], evals])
        vals = np.concatenate([[vals[0]], vals])
    else:
        evals[0] = 0.0

    idx = np.searchsorted(evals, grid, side="right") - 1
    idx = np.clip(idx, 0, vals.size - 1)
    return vals[idx]


def plot_2panel_budget_and_gain(
    path: Path,
    budget_grid: np.ndarray,
    best_id_grid: np.ndarray,
    best_fd_grid: np.ndarray,
    auc_gain: np.ndarray,
    m_edges: np.ndarray,
    budget_evals: float,
):
    """
    Two-panel, paper-ready figure:
      (a) mean±stderr best-so-far / J* vs evaluation budget (step-interp grid)
      (b) per-instance ΔAUC_B vs |E|, with summary annotation

    best_*_grid: (N, G)
    """
    set_pub_style(grid=False)
    b = np.asarray(budget_grid, float)

    mu_id, se_id = _mean_stderr(best_id_grid, axis=0)
    mu_fd, se_fd = _mean_stderr(best_fd_grid, axis=0)

    fig, axs = plt.subplots(1, 2, figsize=(FULL_W, H_TWO), constrained_layout=True)

    # (a) Budget curves
    ax = axs[0]
    ax.plot(b, mu_id, color=COLORS["ID"], label="VQE + ID")
    ax.fill_between(b, mu_id - se_id, mu_id + se_id, color=COLORS["ID"], alpha=0.18, linewidth=0)
    ax.plot(b, mu_fd, color=COLORS["FD"], ls="--", label=r"VQE + BB-FD (SPSA on $F$)")
    ax.fill_between(b, mu_fd - se_fd, mu_fd + se_fd, color=COLORS["FD"], alpha=0.18, linewidth=0)
    ax.set_xlabel("Energy evaluations")
    ax.set_ylabel(r"Best-so-far $\hat F / J^*$")
    ax.set_xlim(float(b[0]), float(b[-1]))
    y_all = np.concatenate([mu_id, mu_fd])
    y0 = max(0.0, float(np.nanmin(y_all) - 0.04))
    y1 = min(1.05, float(np.nanmax(y_all) + 0.04))
    ax.set_ylim(y0, y1)
    ax.legend(loc="lower right", frameon=False)
    ax.text(0.02, 0.98, "(a)", transform=ax.transAxes, ha="left", va="top")

    # (b) ΔAUC vs |E|
    ax = axs[1]
    x = np.asarray(m_edges, float)
    y = np.asarray(auc_gain, float)

    ax.axhline(0.0, color=COLORS["GT"], lw=1.0, ls=":")
    ax.scatter(x, y, s=22, color="#666666", alpha=0.75, edgecolors="none")

    # summary stats
    y_fin = y[np.isfinite(y)]
    if y_fin.size:
        mu = float(np.mean(y_fin))
        sem = float(np.std(y_fin, ddof=1) / math.sqrt(max(1, y_fin.size))) if y_fin.size > 1 else float("nan")
        win = float(np.mean(y_fin > 0.0))
    else:
        mu, sem, win = float("nan"), float("nan"), float("nan")

    # Put a summary marker at the median |E| to avoid clutter.
    x0 = float(np.nanmedian(x)) if np.isfinite(x).any() else 0.0
    ax.errorbar([x0], [mu], yerr=[sem], fmt="o", ms=6,
                color=COLORS["ID"], ecolor=COLORS["ID"], capsize=2, lw=1.2)

    txt = (fr"$B$={budget_evals:.0f} evals"
           "\n"
           fr"$\Delta \mathrm{{AUC}}_B$ mean = {mu:+.4f}"
           + (fr"$\pm${sem:.4f}" if np.isfinite(sem) else "")
           + "\n"
           fr"win rate = {100.0*win:.1f}\%")

    ax.text(
        0.03, 0.97, txt,
        transform=ax.transAxes, ha="left", va="top", fontsize=7,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.85)
    )

    ax.set_xlabel(r"Number of edges $|E|$")
    ax.set_ylabel(r"$\Delta \mathrm{AUC}_B$ (ID $-$ BB-FD)")
    ax.text(0.02, 0.98, "(b)", transform=ax.transAxes, ha="left", va="top")

    _savefig(fig, path)
    plt.close(fig)


def plot_steps_bar(
    path: Path,
    steps_id: np.ndarray,
    steps_fd: np.ndarray,
):
    """
    Bar chart: outer steps completed within budget B (mean±stderr).
    """
    set_pub_style(grid=False)
    fig, ax = plt.subplots(figsize=(COL_W, H_TWO), constrained_layout=True)

    sid = np.asarray(steps_id, float)
    sfd = np.asarray(steps_fd, float)

    mu_id = float(np.nanmean(sid))
    mu_fd = float(np.nanmean(sfd))
    se_id = float(np.nanstd(sid, ddof=1) / math.sqrt(max(1, np.sum(np.isfinite(sid))))) if np.sum(np.isfinite(sid)) > 1 else float("nan")
    se_fd = float(np.nanstd(sfd, ddof=1) / math.sqrt(max(1, np.sum(np.isfinite(sfd))))) if np.sum(np.isfinite(sfd)) > 1 else float("nan")

    xs = np.array([0, 1], int)
    mus = np.array([mu_id, mu_fd], float)
    ses = np.array([se_id, se_fd], float)

    ax.bar(xs, mus, yerr=ses, capsize=3, width=0.6,
           color=["#dddddd", "#dddddd"], edgecolor=["#444444", "#444444"])
    ax.scatter([0], [mu_id], color=COLORS["ID"], s=26, zorder=5, edgecolors="white", linewidth=0.5)
    ax.scatter([1], [mu_fd], color=COLORS["FD"], s=26, zorder=5, edgecolors="white", linewidth=0.5)

    ax.set_xticks(xs)
    ax.set_xticklabels(["ID", "BB-FD"])
    ax.set_ylabel(r"Outer steps within budget $B$")
    ax.set_xlim(-0.6, 1.6)

    ratio = mu_id / max(1e-12, mu_fd)
    ax.text(0.5, 0.98, fr"steps ratio $\approx$ {ratio:.2f}$\times$",
            transform=ax.transAxes, ha="center", va="top", fontsize=7)

    _savefig(fig, path)
    plt.close(fig)


# ==============================================================================
# 2) Utilities: instance generation and bitstring precomputations
# ==============================================================================

def to_uint_seed(seed: int) -> int:
    return int(seed) % (2**32 - 1)


def generate_random_graph(n: int, p_edge: float, rng: np.random.Generator):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p_edge:
                edges.append((i, j))
    return edges


def precompute_z_big_endian(n: int) -> np.ndarray:
    """
    Z[q, state] ∈ {+1,-1} for big-endian ordering.
    """
    K = 1 << n
    idx = np.arange(K, dtype=np.uint32)
    Z = np.empty((n, K), dtype=np.int8)
    for q in range(n):
        bitpos = n - 1 - q
        Z[q] = 1 - 2 * ((idx >> bitpos) & 1).astype(np.int8)
    return Z


def build_cut_mask(edges, Z: np.ndarray) -> np.ndarray:
    """
    cut_mask[state, e] = (1 - Z_i Z_j)/2 ∈ {0,1}
    """
    _, K = Z.shape
    m = len(edges)
    cut = np.empty((K, m), dtype=np.float64)
    for e, (i, j) in enumerate(edges):
        cut[:, e] = 0.5 * (1.0 - (Z[i] * Z[j]).astype(np.float64))
    return cut


def build_ZZ_edges(edges, Z: np.ndarray) -> np.ndarray:
    """
    ZZ[e, state] = Z_i(state) * Z_j(state) ∈ {+1,-1}
    """
    _, K = Z.shape
    m = len(edges)
    ZZ = np.empty((m, K), dtype=np.int8)
    for e, (i, j) in enumerate(edges):
        ZZ[e] = (Z[i] * Z[j]).astype(np.int8)
    return ZZ


# ==============================================================================
# 3) Edge-wise parametric family: w_e(λ_e)
# ==============================================================================

class FamilyEdgeWise:
    """
    Edge-wise parametric coupling family.

    For each edge e:
      w_e(λ_e) = wbar_e + A_e * f_e(x(λ_e))
    with x(λ) = 2(λ - mid)/Δ ∈ [-1,1].

    Canonical shapes (mean-zero, unit-RMS under x~Unif[-1,1]):
      linear:     f = √3 s x
      quadratic:  f = √(45/4) s (x^2 - 1/3)
      periodic:   f = √2 cos(π k x + φ),    k∈{1,...,K}

    Important: This class is "edge-wise": λ is a vector of shape (m,).
    """
    def __init__(self, m: int, kind: str, lam_bounds: Tuple[float, float],
                 rng: np.random.Generator, K: int = 6):
        self.kind = str(kind)
        self.lam_min, self.lam_max = map(float, lam_bounds)
        self.mid = 0.5 * (self.lam_min + self.lam_max)
        self.Delta = max(1e-12, self.lam_max - self.lam_min)
        self.dx_dlam = 2.0 / self.Delta

        # positive baseline + moderate amplitude
        self.wbar = rng.uniform(2.0, 3.0, size=m).astype(float)
        self.A = rng.uniform(0.3, 0.8, size=m).astype(float)

        if self.kind in ("linear", "quadratic"):
            self.s = rng.choice([-1.0, +1.0], size=m).astype(float)
            self.k = None
            self.phi = None
        elif self.kind == "periodic":
            self.s = None
            self.k = rng.integers(1, K + 1, size=m).astype(float)
            self.phi = rng.uniform(0.0, 2*np.pi, size=m).astype(float)
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
            f = c * self.s * (x*x - 1.0/3.0)
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
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        return w.astype(float)

    def dw_dlam(self, lam_vec: np.ndarray) -> np.ndarray:
        _, df = self.f_df(self.x(lam_vec))
        dw = self.A * df * self.dx_dlam
        dw = np.nan_to_num(dw, nan=0.0, posinf=0.0, neginf=0.0)
        return dw.astype(float)

    def w_max(self, grid_points: int = 801) -> np.ndarray:
        """
        Per-edge maximum weight w_e^max over λ ∈ [λmin, λmax], found by grid scan.
        This is used only for diagnostic normalization J*.
        """
        lams = np.linspace(self.lam_min, self.lam_max, int(grid_points), dtype=float)
        x = 2.0 * (lams - self.mid) / self.Delta  # shape (G,)

        if self.kind == "linear":
            c = math.sqrt(3.0)
            # f_e(x) = c*s_e*x -> max over x in [-1,1] is c*|s| = c
            f_max = c * np.ones_like(self.wbar)
        elif self.kind == "quadratic":
            # safer to grid over x: x∈[-1,1]
            X = x[None, :]  # (1,G)
            c = math.sqrt(45.0 / 4.0)
            f_grid = c * self.s[:, None] * (X*X - 1.0/3.0)  # (m,G)
            f_max = np.max(f_grid, axis=1)
        else:
            # periodic: grid scan in x-space (covers full periods because x∈[-1,1])
            X = x[None, :]  # (1,G)
            c = math.sqrt(2.0)
            arg = math.pi * self.k[:, None] * X + self.phi[:, None]
            f_grid = c * np.cos(arg)  # (m,G)
            f_max = np.max(f_grid, axis=1)

        wmax = self.wbar + self.A * f_max
        wmax = np.nan_to_num(wmax, nan=0.0, posinf=0.0, neginf=0.0)
        return wmax.astype(float)


def classical_Jstar_from_wmax(cut_mask: np.ndarray, wmax: np.ndarray) -> float:
    """
    J* = max_z (cut_mask[z] @ wmax), enumerating all 2^n bitstrings.
    """
    with np.errstate(all="ignore"):
        vals = cut_mask @ wmax.astype(np.float64)
    vals = np.nan_to_num(vals, nan=-1e30, posinf=-1e30, neginf=-1e30)
    Jstar = float(np.max(vals))
    if (not np.isfinite(Jstar)) or Jstar <= 0.0:
        Jstar = 1.0
    return Jstar


# ==============================================================================
# 4) VQE simulator (statevector) + SPSA inner solver
# ==============================================================================

CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]], dtype=np.complex128)


def _renorm(psi: np.ndarray) -> np.ndarray:
    nrm = float(np.vdot(psi, psi).real)
    if (not np.isfinite(nrm)) or nrm <= 0:
        psi[:] = 1.0 / np.sqrt(psi.size)
    else:
        psi /= math.sqrt(nrm)
    return psi


def _apply_1q(psi: np.ndarray, gate: np.ndarray, target: int, n: int) -> np.ndarray:
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


def vqe_state(n: int, params: np.ndarray, L: int) -> np.ndarray:
    K = 1 << n
    psi = np.zeros(K, dtype=np.complex128)
    psi[0] = 1.0
    for l in range(L):
        ry = params[l*(2*n): l*(2*n) + n]
        rz = params[l*(2*n) + n: (l+1)*(2*n)]
        for q in range(n):
            cy, sy = math.cos(float(ry[q]) / 2.0), math.sin(float(ry[q]) / 2.0)
            RY = np.array([[cy, -sy], [sy, cy]], dtype=np.complex128)
            psi = _apply_1q(psi, RY, q, n)
            cz, sz = np.exp(-0.5j * float(rz[q])), np.exp(+0.5j * float(rz[q]))
            RZ = np.array([[cz, 0], [0, sz]], dtype=np.complex128)
            psi = _apply_1q(psi, RZ, q, n)
        for q in range(n):
            psi = _apply_2q(psi, CNOT, q, (q + 1) % n, n)
        psi = _renorm(psi)
    return psi


def vqe_expect(
    n: int,
    params: np.ndarray,
    L: int,
    w: np.ndarray,
    ZZ_edges: np.ndarray,
    cut_mask: np.ndarray,
    shots: int,
    rng: np.random.Generator,
):
    """
    Returns:
      J (float), p_cut (m,), psi (statevector)
    If shots>0, p_cut is estimated from bitstring samples.
    """
    psi = vqe_state(n, params, L)
    probs = (psi.conj() * psi).real.astype(np.float64)
    s = float(np.sum(probs))
    if (not np.isfinite(s)) or s <= 0:
        probs[:] = 1.0 / probs.size
    else:
        probs /= s
    np.nan_to_num(probs, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    if shots > 0:
        idx = rng.choice(np.arange(probs.size), size=shots, replace=True, p=probs)
        with np.errstate(all="ignore"):
            p_cut = np.mean(cut_mask[idx, :], axis=0).astype(np.float64)
        np.nan_to_num(p_cut, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        with np.errstate(all="ignore"):
            zexp = (ZZ_edges.astype(np.float64) @ probs).astype(np.float64)
        np.nan_to_num(zexp, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        zexp = np.clip(zexp, -1.0, 1.0)
        p_cut = 0.5 * (1.0 - zexp)

    with np.errstate(all="ignore"):
        J = float(np.dot(p_cut, w))
    if not np.isfinite(J):
        J = 0.0

    return J, p_cut, psi


def vqe_energy(
    n: int,
    params: np.ndarray,
    L: int,
    w: np.ndarray,
    ZZ_edges: np.ndarray,
    cut_mask: np.ndarray,
    shots: int,
    rng: np.random.Generator,
) -> float:
    J, _, _ = vqe_expect(n, params, L, w, ZZ_edges, cut_mask, shots, rng)
    return -J


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
):
    """
    Returns: best_p, best_E, evals_used
    Per SPSA iter: Ep, Em, E -> 3 energy evaluations.
    """
    rng = np.random.default_rng(to_uint_seed(seed))
    p = p0.astype(float).copy()
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)

    best_p = p.copy()
    best_E = float("inf")
    evals = 0

    for k in range(1, iters + 1):
        ak = a / ((k + A) ** alpha)
        ck = c / (k ** gamma)
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


def inner_solve(
    n: int,
    L: int,
    w: np.ndarray,
    ZZ_edges: np.ndarray,
    cut_mask: np.ndarray,
    init_params: np.ndarray,
    inner_iters: int,
    restarts: int,
    seed_base: int,
    shots: int,
):
    """
    Run 'restarts' SPSA inner solves and pick the best (lowest energy).
    Returns:
      best_params, best_J, best_p_cut, evals_used
    """
    D = init_params.size
    bounds = [(-math.pi, math.pi)] * D

    best_params = init_params.copy()
    best_E = float("inf")
    evals_total = 0

    for r in range(int(restarts)):
        # restart init
        if r == 0:
            p0 = init_params.copy()
        else:
            rrng = np.random.default_rng(to_uint_seed(seed_base + 999 * (r + 1)))
            p0 = rrng.uniform(-math.pi, math.pi, size=D).astype(float)

        # shot RNG for this inner run (only used if shots>0)
        eval_rng = np.random.default_rng(to_uint_seed(seed_base + 1337 * (r + 1)))

        def Efun(pvec):
            return vqe_energy(n, pvec, L, w, ZZ_edges, cut_mask, shots, eval_rng)

        p_star, E_star, ev = spsa_minimize(
            Efun, p0, bounds, iters=inner_iters,
            seed=seed_base + 17 * (r + 1),
        )
        evals_total += int(ev)

        if np.isfinite(E_star) and E_star < best_E:
            best_E = E_star
            best_params = p_star.copy()

    # evaluate final best to obtain J and p_cut (counts as ONE energy evaluation)
    eval_rng_final = np.random.default_rng(to_uint_seed(seed_base + 424242))
    J, p_cut, _ = vqe_expect(n, best_params, L, w, ZZ_edges, cut_mask, shots, eval_rng_final)
    evals_total += 1

    return best_params, float(J), p_cut.astype(np.float64), int(evals_total)


# ==============================================================================
# 5) Outer loops (edge-wise λ) under evaluation budget B
# ==============================================================================

def auc_step(evals: np.ndarray, values: np.ndarray, budget: float) -> float:
    """
    Step-function AUC on [0, budget].
    Assumes best-so-far is held constant between recorded events.
    """
    evals = np.asarray(evals, float)
    values = np.asarray(values, float)
    budget = float(budget)
    if evals.size == 0:
        return 0.0

    # prepend at 0 with first value
    if evals[0] > 0.0:
        evals = np.insert(evals, 0, 0.0)
        values = np.insert(values, 0, float(values[0]))

    # truncate to <= budget and append exactly budget
    m = evals <= budget
    evals = evals[m]
    values = values[m]
    if evals.size == 0:
        return float(values[0])

    if evals[-1] < budget:
        evals = np.append(evals, budget)
        values = np.append(values, values[-1])

    dx = np.diff(evals)
    area = float(np.sum(dx * values[:-1]))
    return area / float(budget)


def _best_at_budget(evals: np.ndarray, best_norm: np.ndarray, budget: float) -> float:
    g = np.array([float(budget)], float)
    return float(_step_interp(evals, best_norm, g)[0])


def run_outer_budget_edgewise(
    mode: str,
    n: int,
    edges,
    fam: FamilyEdgeWise,
    ZZ_edges: np.ndarray,
    cut_mask: np.ndarray,
    lam_vec0: np.ndarray,
    outer_max: int,
    inner_iters: int,
    restarts: int,
    L: int,
    seed: int,
    eta0: float,
    eta_pow: float,
    step_clip: float,
    c_frac: float,
    shots: int,
    budget_evals: float,
    Jstar: float,
    fd_warmstart: str = "best",
) -> Dict[str, np.ndarray]:
    """
    Runs outer optimization until budget or outer_max.

    Returns trace dictionaries:
      events_evals: cumulative evals after each value-query completion (center, and +/− for FD)
      events_best : best-so-far normalized by J* at each event
      outer_evals : cumulative evals after each completed OUTER iteration (after update)
      evals_end   : total evals used
      lam_vec_end : final λ vector
    """
    lam_min, lam_max = fam.lam_min, fam.lam_max
    lam_vec = np.clip(np.asarray(lam_vec0, float), lam_min, lam_max)

    D = 2 * n * L
    params = np.zeros(D, float)

    evals = 0.0
    best = -1e30

    events_evals: List[float] = []
    events_best: List[float] = []
    outer_evals: List[float] = []

    c = float(c_frac * (lam_max - lam_min))

    rng_dir = np.random.default_rng(to_uint_seed(seed + 999999))

    for t in range(1, int(outer_max) + 1):
        # --- center value query F(λ) via inner solve
        w = fam.w(lam_vec)

        params_c, Jc, p_cut_c, ev_c = inner_solve(
            n=n, L=L, w=w, ZZ_edges=ZZ_edges, cut_mask=cut_mask,
            init_params=params, inner_iters=inner_iters, restarts=restarts,
            seed_base=seed + 100000 * t + 111, shots=shots
        )
        params = params_c
        evals += float(ev_c)

        best = max(best, float(Jc))
        events_evals.append(evals)
        events_best.append(best / Jstar)

        if mode == "ID":
            # full hypergradient vector (correlator reuse): g_e = w'_e(λ_e) * p_e
            g_vec = fam.dw_dlam(lam_vec) * p_cut_c

        elif mode == "FD_SPSA":
            # random direction Δ ∈ {±1}^m
            delta = rng_dir.choice([-1.0, +1.0], size=lam_vec.size).astype(float)

            lam_p = np.clip(lam_vec + c * delta, lam_min, lam_max)
            lam_m = np.clip(lam_vec - c * delta, lam_min, lam_max)

            # + perturbation solve
            w_p = fam.w(lam_p)
            params_p, Jp, _, ev_p = inner_solve(
                n=n, L=L, w=w_p, ZZ_edges=ZZ_edges, cut_mask=cut_mask,
                init_params=params, inner_iters=inner_iters, restarts=restarts,
                seed_base=seed + 100000 * t + 777, shots=shots
            )
            evals += float(ev_p)
            best = max(best, float(Jp))
            events_evals.append(evals)
            events_best.append(best / Jstar)

            # - perturbation solve
            w_m = fam.w(lam_m)
            params_m, Jm, _, ev_m = inner_solve(
                n=n, L=L, w=w_m, ZZ_edges=ZZ_edges, cut_mask=cut_mask,
                init_params=params, inner_iters=inner_iters, restarts=restarts,
                seed_base=seed + 100000 * t + 999, shots=shots
            )
            evals += float(ev_m)
            best = max(best, float(Jm))
            events_evals.append(evals)
            events_best.append(best / Jstar)

            # SPSA-FD gradient estimate on F
            g_scalar = (float(Jp) - float(Jm)) / (2.0 * c) if c > 0 else 0.0
            g_vec = g_scalar * delta

            # optional warm-start for the NEXT center solve
            if fd_warmstart == "best":
                # pick params from the best of {center, plus, minus}
                if (Jp >= Jc) and (Jp >= Jm):
                    params = params_p
                elif (Jm >= Jc) and (Jm >= Jp):
                    params = params_m
                else:
                    params = params_c
            else:
                # keep center params (default earlier scripts)
                params = params_c

        else:
            raise ValueError("mode must be 'ID' or 'FD_SPSA'")

        # --- outer update (projected gradient ascent on box)
        eta = float(eta0 / (t ** eta_pow))
        step = eta * g_vec
        step = np.clip(step, -float(step_clip), float(step_clip))
        lam_vec = np.clip(lam_vec + step, lam_min, lam_max)

        outer_evals.append(float(evals))

        if evals >= float(budget_evals):
            break

    return {
        "events_evals": np.asarray(events_evals, float),
        "events_best": np.asarray(events_best, float),
        "outer_evals": np.asarray(outer_evals, float),
        "evals_end": np.array([evals], float),
        "lam_vec_end": lam_vec.astype(float),
    }


def steps_within_budget(outer_evals: np.ndarray, budget: float) -> int:
    outer_evals = np.asarray(outer_evals, float)
    return int(np.sum(outer_evals <= float(budget)))


# ==============================================================================
# 6) CSV / table writers
# ==============================================================================

def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_table_tex(path: Path, summary_rows: List[Dict], caption: str, label: str):
    """
    Minimal booktabs LaTeX table:
      metric | ID mean±stderr | FD mean±stderr
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("% Auto-generated by exp7_edgewise_lambda_vector.py\n")
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\toprule\n")
        f.write("Metric & VQE+ID & VQE+BB-FD (SPSA on $F$)\\\\\n")
        f.write("\\midrule\n")
        for r in summary_rows:
            f.write(f"{r['metric']} & {r['id_mean']:.3f}$\\pm${r['id_se']:.3f} & "
                    f"{r['fd_mean']:.3f}$\\pm${r['fd_se']:.3f}\\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


# ==============================================================================
# 7) Main
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="out_exp7_edgewise")
    p.add_argument("--fmt", type=str, default="pdf", choices=["pdf", "png", "svg"])

    # instances / seeds
    p.add_argument("--seed0", type=int, default=7)
    p.add_argument("--num_instances", type=int, default=20)

    # problem
    p.add_argument("--family", type=str, default="periodic", choices=["linear", "quadratic", "periodic"])
    p.add_argument("--periodic_K", type=int, default=6)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--p_edge", type=float, default=0.45)
    p.add_argument("--lam_min", type=float, default=-5.0)
    p.add_argument("--lam_max", type=float, default=5.0)
    p.add_argument("--lam0", type=float, default=0.8)

    # VQE / inner
    p.add_argument("--L", type=int, default=2)
    p.add_argument("--inner_iters", type=int, default=10)
    p.add_argument("--restarts", type=int, default=1)
    p.add_argument("--shots", type=int, default=0, help="shots per energy evaluation (0 = exact)")

    # outer
    p.add_argument("--outer_max", type=int, default=400)
    p.add_argument("--eta0", type=float, default=0.35)
    p.add_argument("--eta_pow", type=float, default=0.6)
    p.add_argument("--step_clip", type=float, default=0.6)
    p.add_argument("--c_frac", type=float, default=0.05)

    # budget + normalization scan
    p.add_argument("--budget_evals", type=float, default=5000)
    p.add_argument("--wmax_grid", type=int, default=801, help="grid points to scan per-edge w^max for J* diagnostic")
    p.add_argument("--budget_points", type=int, default=240, help="points in shared budget grid for plotting")

    # FD warm start
    p.add_argument("--fd_warmstart", type=str, default="best", choices=["best", "center"],
                   help="How to warm-start the next center solve in FD. 'best' uses the best of center/+/- params.")

    return p.parse_args()


def main():
    a = parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # Precompute Z once (depends only on n)
    Z = precompute_z_big_endian(a.n)

    run_rows: List[Dict] = []

    # store per-instance budget-interpolated traces (best-so-far / J*)
    budget_grid = np.linspace(0.0, float(a.budget_evals), int(a.budget_points))
    best_id_grid_list: List[np.ndarray] = []
    best_fd_grid_list: List[np.ndarray] = []

    # per-instance scalars
    auc_id_list = []
    auc_fd_list = []
    auc_gain_list = []
    steps_id_list = []
    steps_fd_list = []
    m_edges_list = []

    for r in range(int(a.num_instances)):
        seed = int(a.seed0 + r)
        rng = np.random.default_rng(to_uint_seed(seed))

        edges = generate_random_graph(a.n, a.p_edge, rng)
        if not edges:
            # deterministic bump to avoid empty graphs
            rng2 = np.random.default_rng(to_uint_seed(seed + 12345))
            edges = generate_random_graph(a.n, a.p_edge, rng2)
        if not edges:
            continue

        m = len(edges)
        cut_mask = build_cut_mask(edges, Z)
        ZZ_edges = build_ZZ_edges(edges, Z)

        fam = FamilyEdgeWise(m=m, kind=a.family, lam_bounds=(a.lam_min, a.lam_max), rng=rng, K=a.periodic_K)

        # diagnostic normalization J*
        wmax = fam.w_max(grid_points=a.wmax_grid)
        Jstar = classical_Jstar_from_wmax(cut_mask, wmax)
        if (not np.isfinite(Jstar)) or Jstar <= 0:
            Jstar = 1.0

        lam_vec0 = np.full(m, float(a.lam0), dtype=float)

        # ID run
        hist_id = run_outer_budget_edgewise(
            mode="ID",
            n=a.n, edges=edges, fam=fam, ZZ_edges=ZZ_edges, cut_mask=cut_mask,
            lam_vec0=lam_vec0,
            outer_max=a.outer_max,
            inner_iters=a.inner_iters,
            restarts=a.restarts,
            L=a.L,
            seed=seed,
            eta0=a.eta0,
            eta_pow=a.eta_pow,
            step_clip=a.step_clip,
            c_frac=a.c_frac,
            shots=a.shots,
            budget_evals=a.budget_evals,
            Jstar=Jstar,
            fd_warmstart=a.fd_warmstart,
        )

        # FD run (SPSA finite difference on F)
        hist_fd = run_outer_budget_edgewise(
            mode="FD_SPSA",
            n=a.n, edges=edges, fam=fam, ZZ_edges=ZZ_edges, cut_mask=cut_mask,
            lam_vec0=lam_vec0,
            outer_max=a.outer_max,
            inner_iters=a.inner_iters,
            restarts=a.restarts,
            L=a.L,
            seed=seed,
            eta0=a.eta0,
            eta_pow=a.eta_pow,
            step_clip=a.step_clip,
            c_frac=a.c_frac,
            shots=a.shots,
            budget_evals=a.budget_evals,
            Jstar=Jstar,
            fd_warmstart=a.fd_warmstart,
        )

        # Interpolate best-so-far onto shared budget grid
        best_id_grid = _step_interp(hist_id["events_evals"], hist_id["events_best"], budget_grid)
        best_fd_grid = _step_interp(hist_fd["events_evals"], hist_fd["events_best"], budget_grid)
        best_id_grid_list.append(best_id_grid)
        best_fd_grid_list.append(best_fd_grid)

        # Scalars
        auc_id = auc_step(hist_id["events_evals"], hist_id["events_best"], a.budget_evals)
        auc_fd = auc_step(hist_fd["events_evals"], hist_fd["events_best"], a.budget_evals)
        gain = float(auc_id - auc_fd)

        best_final_id = _best_at_budget(hist_id["events_evals"], hist_id["events_best"], a.budget_evals)
        best_final_fd = _best_at_budget(hist_fd["events_evals"], hist_fd["events_best"], a.budget_evals)

        steps_id = steps_within_budget(hist_id["outer_evals"], a.budget_evals)
        steps_fd = steps_within_budget(hist_fd["outer_evals"], a.budget_evals)

        auc_id_list.append(auc_id)
        auc_fd_list.append(auc_fd)
        auc_gain_list.append(gain)
        steps_id_list.append(steps_id)
        steps_fd_list.append(steps_fd)
        m_edges_list.append(m)

        run_rows.append({
            "instance": r,
            "seed": seed,
            "family": a.family,
            "K": float(a.periodic_K) if a.family == "periodic" else float("nan"),
            "n": float(a.n),
            "p_edge": float(a.p_edge),
            "m_edges": int(m),
            "L": int(a.L),
            "inner_iters": int(a.inner_iters),
            "restarts": int(a.restarts),
            "shots": int(a.shots),
            "budget_evals": float(a.budget_evals),
            "c_frac": float(a.c_frac),
            "eta0": float(a.eta0),
            "eta_pow": float(a.eta_pow),
            "step_clip": float(a.step_clip),
            "fd_warmstart": a.fd_warmstart,
            "Jstar": float(Jstar),

            "auc_id": float(auc_id),
            "auc_fd": float(auc_fd),
            "auc_gain": float(gain),

            "best_final_id": float(best_final_id),
            "best_final_fd": float(best_final_fd),

            "steps_id": int(steps_id),
            "steps_fd": int(steps_fd),

            "evals_end_id": float(hist_id["evals_end"][0]),
            "evals_end_fd": float(hist_fd["evals_end"][0]),
        })

        print(f"[inst={r:02d} seed={seed:3d} |m|={m:2d}] "
              f"AUC_ID={auc_id:.4f} AUC_FD={auc_fd:.4f} gain={gain:+.4f}  "
              f"steps(ID,FD)=({steps_id},{steps_fd})  "
              f"final(ID,FD)=({best_final_id:.3f},{best_final_fd:.3f})")

    if not run_rows:
        raise RuntimeError("No instances generated (graphs had 0 edges). Try increasing p_edge or changing seed0.")

    # Save per-instance CSV
    runs_csv = out / "runs7_edgewise_metrics.csv"
    write_csv(runs_csv, run_rows, fieldnames=list(run_rows[0].keys()))

    # Stack for plotting
    best_id_grid_all = np.vstack(best_id_grid_list)
    best_fd_grid_all = np.vstack(best_fd_grid_list)
    auc_gain = np.asarray(auc_gain_list, float)
    m_edges = np.asarray(m_edges_list, float)
    steps_id = np.asarray(steps_id_list, float)
    steps_fd = np.asarray(steps_fd_list, float)

    N = best_id_grid_all.shape[0]
    suf = (f"{a.family}_n{a.n}_B{int(a.budget_evals)}_inner{a.inner_iters}_R{a.restarts}_"
           f"S{a.shots}_seed0{a.seed0}_N{N}")

    # Figure(s)
    fig_path = out / f"fig7_edgewise_budget_gain_{suf}.{a.fmt}"
    plot_2panel_budget_and_gain(
        fig_path,
        budget_grid=budget_grid,
        best_id_grid=best_id_grid_all,
        best_fd_grid=best_fd_grid_all,
        auc_gain=auc_gain,
        m_edges=m_edges,
        budget_evals=float(a.budget_evals),
    )

    fig_steps = out / f"fig7_edgewise_steps_{suf}.{a.fmt}"
    plot_steps_bar(fig_steps, steps_id=steps_id, steps_fd=steps_fd)

    # Summary table (mean ± stderr)
    def _summ(arr: np.ndarray):
        arr = np.asarray(arr, float)
        mu = float(np.nanmean(arr))
        se = float(np.nanstd(arr, ddof=1) / math.sqrt(max(1, np.sum(np.isfinite(arr))))) if np.sum(np.isfinite(arr)) > 1 else float("nan")
        return mu, se

    best_final_id = np.array([row["best_final_id"] for row in run_rows], float)
    best_final_fd = np.array([row["best_final_fd"] for row in run_rows], float)
    auc_id = np.array([row["auc_id"] for row in run_rows], float)
    auc_fd = np.array([row["auc_fd"] for row in run_rows], float)
    steps_id_arr = np.array([row["steps_id"] for row in run_rows], float)
    steps_fd_arr = np.array([row["steps_fd"] for row in run_rows], float)

    summary_rows = []
    for metric, arr_id, arr_fd in [
        (r"Final best-so-far / $J^*$", best_final_id, best_final_fd),
        (r"$\mathrm{AUC}_B$", auc_id, auc_fd),
        (r"Outer steps within budget $B$", steps_id_arr, steps_fd_arr),
    ]:
        idm, ids = _summ(arr_id)
        fdm, fds = _summ(arr_fd)
        summary_rows.append({"metric": metric, "id_mean": idm, "id_se": ids, "fd_mean": fdm, "fd_se": fds})

    # Save summary CSV
    table_csv = out / "table7_edgewise_summary.csv"
    with table_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "id_mean", "id_stderr", "fd_mean", "fd_stderr"])
        w.writeheader()
        for rrow in summary_rows:
            w.writerow({
                "metric": rrow["metric"],
                "id_mean": f"{rrow['id_mean']:.6f}",
                "id_stderr": f"{rrow['id_se']:.6f}",
                "fd_mean": f"{rrow['fd_mean']:.6f}",
                "fd_stderr": f"{rrow['fd_se']:.6f}",
            })

    table_tex = out / "table7_edgewise_summary.tex"
    write_table_tex(
        table_tex,
        summary_rows,
        caption=(f"Experiment 7 (edge-wise outer parameters). "
                 f"Mean$\\pm$stderr over $N={N}$ random instances for a shared evaluation budget "
                 f"$B={int(a.budget_evals)}$ (energy evaluations)."),
        label="tab:exp7_edgewise",
    )

    # Compact text summary
    txt = out / "exp7_edgewise_summary.txt"
    with txt.open("w", encoding="utf-8") as f:
        f.write("Experiment 7 — Edge-wise outer parameters (vector λ)\n")
        f.write(f"family={a.family} | K={a.periodic_K if a.family=='periodic' else 'NA'} | n={a.n} | p_edge={a.p_edge}\n")
        f.write(f"inner_iters={a.inner_iters} | restarts={a.restarts} | L={a.L} | shots={a.shots}\n")
        f.write(f"budget_evals={a.budget_evals} | num_instances={N} | seed0={a.seed0}\n")
        f.write(f"FD warm-start: {a.fd_warmstart}\n\n")

        mu_gain = float(np.mean(auc_gain[np.isfinite(auc_gain)]))
        sem_gain = float(np.std(auc_gain[np.isfinite(auc_gain)], ddof=1) / math.sqrt(max(1, np.sum(np.isfinite(auc_gain))))) if np.sum(np.isfinite(auc_gain)) > 1 else float("nan")
        win = float(np.mean(auc_gain[np.isfinite(auc_gain)] > 0.0))

        f.write(f"ΔAUC_B mean (ID−FD): {mu_gain:+.6f}  (sem={sem_gain:.6f})\n")
        f.write(f"win rate (ΔAUC>0): {100.0*win:.2f}%\n\n")

        for rrow in summary_rows:
            f.write(f"{rrow['metric']}: ID={rrow['id_mean']:.4f}±{rrow['id_se']:.4f} | "
                    f"FD={rrow['fd_mean']:.4f}±{rrow['fd_se']:.4f}\n")

        f.write("\nFiles:\n")
        f.write(f"  Figure (budget+gain): {fig_path.name}\n")
        f.write(f"  Figure (steps):       {fig_steps.name}\n")
        f.write(f"  Runs CSV:             {runs_csv.name}\n")
        f.write(f"  Summary table:        {table_tex.name}\n")

    print("\nSaved to:", out.resolve())
    print("Figure:", fig_path.name)
    print("Steps fig:", fig_steps.name)
    print("Runs CSV:", runs_csv.name)
    print("Summary:", table_tex.name, "/", table_csv.name)
    print("Text:", txt.name)


if __name__ == "__main__":
    main()
