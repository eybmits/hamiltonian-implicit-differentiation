#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp2C_vqe_vs_qaoa_readout_pubplots_upgraded.py
=============================================

Experiment 2C (Readout bridge, upgraded): VQE vs QAOA under a matched outer protocol,
and how expectation improvements translate to sampled solutions
----------------------------------------------------------------------------------

Paper story (why this exists)
  In Exp. 2A we compare bilevel outer-optimization performance across ansätze.
  But in combinatorial optimization we ultimately care about sampling *good bitstrings*
  with a finite readout budget.

  This script is the "readout realism" bridge (cf. your Experiment 3):
    - Track best-so-far expectation value along the outer trajectory
    - Track best-so-far sampled solution quality along the same trajectory
      under a fixed readout shot budget S per outer step.

  Optional (recommended for rebuttal / supplement):
    - Track a tail metric that saturates much more slowly than best-of-256, e.g.
        Hit-rate:   P(sample >= (1-ε) J*)
      or
        Top-k tail: P(sample is in top-k% of bitstrings by cut value).

Correct bilevel / hypergradient choices (minimal but correct)
  We optimize the value function F(λ)=max_ϑ J(ϑ,λ) via an outer update in λ.
  At an inner optimum ϑ*(λ), the envelope theorem gives:
      dF/dλ = ∂_λ J(ϑ*(λ), λ).

  VQE (λ-independent state at fixed parameters)
      |ψ_VQE(ϕ)⟩ does not depend on λ  ⇒  ∂_λ J is exactly the "reuse term"
          ∂_λ J = Σ_e w'_e(λ) p_e(ϕ)
      and can be computed without extra objective probes.

  QAOA (problem-dependent state)
      |ψ_QAOA(θ,λ)⟩ depends on λ via the cost unitary U_C(λ)=exp(-i γ H_C(λ)).
      Therefore ∂_λ J includes an additional state-dependence term.
      Here we use the *safe* option for QAOA:
          ∂_λ J(θ*,λ) ≈ [J(θ*,λ+c) - J(θ*,λ-c)] / (2c),
      i.e. fixed-θ central finite differences (2 extra objective calls per outer step).

What we report (paper-friendly)
  Main figure (single column, two panels; vs ENERGY EVALUATION BUDGET):
    (a) best-so-far expectation (normalized by J*)
    (b) best-so-far sampled cut (best-of-S readout, normalized by J*)

  Optional figures:
    - Tail probability vs budget (hit-rate or top-k)
    - Readout shot sweep at fixed budget B (best-of-S for S in {16,32,64,256,...})
    - Cost plot: cumulative energy evaluations vs outer step
    - Tradeoff scatter at budget B with y=x diagonal:
        "above diagonal" = stronger tail advantage (sampling closes expectation gap)

Outputs (in --out folder)
  - fig2C_expect_and_readout_vs_evals.<fmt>
  - fig2C_tailprob_vs_evals.<fmt>                (if --tail_metric != none)
  - fig2C_readout_shots_sweep.<fmt>              (if --supp_shots not empty)
  - fig2C_evals_vs_outer.<fmt>                   (can disable with --no_cost_plot)
  - fig2C_tradeoff_scatter.<fmt>                 (can disable with --no_tradeoff_plot)
  - exp2C_results.csv
  - exp2C_summary.txt

Example
  python exp2C_vqe_vs_qaoa_readout_pubplots_upgraded.py \
    --kind periodic --periodic_K 6 --seeds 7,8,9,10,11,12,13,14 \
    --readout_shots 256 --supp_shots 16,32,64 \
    --tail_metric hit --hit_eps 0.10 \
    --fmt pdf

Notes
  - Inner solver: SPSA uses 3 objective calls per iteration (Ep, Em, E).
  - Readout shots are NOT counted as "energy evaluations" (budget-first framing).
  - CLEAN PLOTS: no in-plot annotation boxes; only the legend is used to identify methods.
"""

import math
import argparse
import warnings
import logging
import csv
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# ------------------------------------------------------------------------------
# Silence known noisy-but-harmless messages (fontTools uses logging)
# ------------------------------------------------------------------------------
warnings.filterwarnings("ignore", message=".*timestamp seems very low.*")
warnings.filterwarnings("ignore", message=".*regarding as unix timestamp.*")

_ft = logging.getLogger("fontTools")
_ft.setLevel(logging.ERROR)
_ft.propagate = False
if not _ft.handlers:
    _ft.addHandler(logging.NullHandler())
logging.getLogger("fontTools.ttLib").setLevel(logging.ERROR)
logging.getLogger("fontTools.subset").setLevel(logging.ERROR)


# ==============================================================================
# 1) Minimal NPJ/Nature-ish plotting
# ==============================================================================

COLORS = {
    "VQE":  "#D62728",  # red
    "QAOA": "#1F77B4",  # blue
}

COL_W = 3.37   # inches (single-column)
H_COL = 2.8    # inches


def fig_size() -> Tuple[float, float]:
    return (COL_W, H_COL)


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
        "lines.linewidth": 1.5,
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
    plt.close(fig)


def _panel_label(ax: plt.Axes, label: str):
    ax.text(0.00, 1.02, label, transform=ax.transAxes,
            va="bottom", ha="left", fontsize=9, fontweight="bold")


# ==============================================================================
# 2) Utilities
# ==============================================================================

def to_uint_seed(seed: int) -> int:
    return int(seed) % (2**32 - 1)


def parse_int_list(s: str) -> List[int]:
    s = (s or "").strip()
    if not s:
        return []
    out = []
    for chunk in s.split(","):
        chunk = chunk.strip()
        if chunk:
            out.append(int(chunk))
    return out


def step_sample(evals: np.ndarray, y_step: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """
    Sample a step function y(evals) on a grid.
    For grid < evals[0], output 0.
    """
    evals = np.asarray(evals, float)
    y_step = np.asarray(y_step, float)
    grid = np.asarray(grid, float)
    if evals.size == 0:
        return np.zeros_like(grid)
    idx = np.searchsorted(evals, grid, side="right") - 1
    out = np.zeros_like(grid, dtype=float)
    m = idx >= 0
    out[m] = y_step[idx[m]]
    return out


def step_value_at(evals: np.ndarray, y_step: np.ndarray, x: float) -> float:
    evals = np.asarray(evals, float)
    y_step = np.asarray(y_step, float)
    if evals.size == 0:
        return 0.0
    idx = np.searchsorted(evals, float(x), side="right") - 1
    if idx < 0:
        return 0.0
    return float(y_step[idx])


def step_auc(evals: np.ndarray, y_step: np.ndarray, x_max: float) -> float:
    """
    AUC of a step function y(evals) over [0, x_max] (piecewise-constant, right-continuous).
    """
    evals = np.asarray(evals, float)
    y_step = np.asarray(y_step, float)
    x_max = float(x_max)

    if evals.size == 0 or x_max <= 0:
        return 0.0

    area = 0.0
    prev_x = 0.0
    prev_y = 0.0

    for x, y in zip(evals, y_step):
        x = float(min(x, x_max))
        if x <= prev_x:
            prev_y = float(y)
            continue
        area += (x - prev_x) * prev_y
        prev_x = x
        prev_y = float(y)
        if prev_x >= x_max:
            break

    if prev_x < x_max:
        area += (x_max - prev_x) * prev_y
    return float(area)


def write_csv(path: Path, rows: List[Dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ==============================================================================
# 3) Instance generation + parameter family
# ==============================================================================

def generate_random_graph(n: int, p_edge: float, rng: np.random.Generator):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p_edge:
                edges.append((i, j))
    return edges


def precompute_z_big_endian(n: int) -> np.ndarray:
    """Z[q, x] = ±1 eigenvalue of Z on qubit q for computational basis state index x."""
    K = 1 << n
    idx = np.arange(K, dtype=np.uint32)
    Z = np.empty((n, K), dtype=np.int8)
    for q in range(n):
        bitpos = n - 1 - q
        Z[q] = 1 - 2 * ((idx >> bitpos) & 1).astype(np.int8)
    return Z


def build_cut_mask(edges, Z: np.ndarray) -> np.ndarray:
    """
    cut_mask[x, e] = 1 if edge e is cut by bitstring x, else 0.
    """
    K = Z.shape[1]
    m = len(edges)
    cut = np.empty((K, m), dtype=np.float64)
    for e, (i, j) in enumerate(edges):
        cut[:, e] = 0.5 * (1.0 - (Z[i] * Z[j]).astype(np.float64))
    return cut


class Family1D:
    """
    w_e(λ) = w̄_e + A_e f_e(x), x = 2(λ-mid)/Δ ∈ [-1,1]
    with mean-zero RMS-normalized f_e.

    Supported:
      linear, quadratic, periodic (cosine with random phase & frequency).
    """
    def __init__(self, m: int, kind: str, lam_bounds: Tuple[float, float],
                 rng: np.random.Generator, K: int = 6):
        self.kind = str(kind)
        self.lam_min, self.lam_max = map(float, lam_bounds)
        self.mid = 0.5 * (self.lam_min + self.lam_max)
        self.Delta = max(1e-12, self.lam_max - self.lam_min)
        self.dx_dlam = 2.0 / self.Delta

        self.wbar = rng.uniform(2.0, 3.0, size=m).astype(float)
        self.A = rng.uniform(0.3, 0.8, size=m).astype(float)

        if self.kind in ("linear", "quadratic"):
            self.s = rng.choice([-1.0, +1.0], size=m).astype(float)
            self.k = None
            self.phi = None
        elif self.kind == "periodic":
            self.k = rng.integers(1, int(K) + 1, size=m).astype(float)
            self.phi = rng.uniform(0.0, 2 * np.pi, size=m).astype(float)
            self.s = None
        else:
            raise ValueError("kind must be linear|quadratic|periodic")

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
        else:  # periodic
            c = math.sqrt(2.0)
            arg = math.pi * self.k * x + self.phi
            f = c * np.cos(arg)
            df = c * (-math.pi * self.k) * np.sin(arg)
        return f, df

    def w(self, lam: float) -> np.ndarray:
        f, _ = self.f_df(self.x(lam))
        w = self.wbar + self.A * f
        return np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0).astype(float)

    def dw_dlam(self, lam: float) -> np.ndarray:
        _, df = self.f_df(self.x(lam))
        dw = self.A * df * self.dx_dlam
        return np.nan_to_num(dw, nan=0.0, posinf=0.0, neginf=0.0).astype(float)


# ==============================================================================
# 4) Statevector simulator primitives
# ==============================================================================

CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]], dtype=np.complex128)


def _renorm(psi: np.ndarray) -> np.ndarray:
    nrm = float(np.vdot(psi, psi).real)
    if (not np.isfinite(nrm)) or nrm <= 0:
        psi[:] = 1.0 / math.sqrt(psi.size)
    else:
        psi /= math.sqrt(nrm)
    return psi


def _apply_1q(psi: np.ndarray, gate: np.ndarray, target: int, n: int) -> np.ndarray:
    with np.errstate(all="ignore"):
        psi_r = psi.reshape([2] * n)
        psi_r = np.moveaxis(psi_r, target, 0)
        block = psi_r.reshape(2, -1).astype(np.complex128, copy=False)
        out = gate @ block
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
        out = gate4 @ block
        psi_r = out.reshape(2, 2, *psi_r.shape[2:])
        psi = np.moveaxis(psi_r, (0, 1), (a, b)).reshape(-1)
    return psi


# ==============================================================================
# 5) Expectation value via cut-mask (fast)
# ==============================================================================

def probs_from_state(psi: np.ndarray) -> np.ndarray:
    probs = (psi.conj() * psi).real.astype(np.float64)
    s = float(np.sum(probs))
    if (not np.isfinite(s)) or s <= 0:
        probs[:] = 1.0 / probs.size
    else:
        probs /= s
    return np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)


def expect_J(psi: np.ndarray, cut_mask: np.ndarray, w: np.ndarray):
    """
    Returns:
      J      : expected cut value
      p_cut  : vector of edge cut probabilities p_e = E[C_e]
      probs  : computational basis distribution
    """
    probs = probs_from_state(psi)
    p_cut = probs @ cut_mask            # shape (m,)
    J = float(p_cut @ w)
    if not np.isfinite(J):
        J = 0.0
    return float(J), p_cut.astype(np.float64), probs


# ==============================================================================
# 6) VQE ansatz (hardware-efficient, λ-independent at fixed params)
# ==============================================================================

def vqe_state(n: int, params: np.ndarray, L: int) -> np.ndarray:
    K = 1 << n
    psi = np.zeros(K, dtype=np.complex128)
    psi[0] = 1.0
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
        psi = _renorm(psi)
    return psi


def vqe_energy(n: int, cut_mask: np.ndarray, w: np.ndarray, params: np.ndarray, L: int) -> float:
    psi = vqe_state(n, params, L)
    J, _, _ = expect_J(psi, cut_mask, w)
    return -J


# ==============================================================================
# 7) QAOA ansatz (problem-dependent: λ enters via w(λ) in the cost unitary)
# ==============================================================================

def RX(theta: float) -> np.ndarray:
    ct = math.cos(theta / 2)
    st = math.sin(theta / 2)
    return np.array([[ct, -1j * st],
                     [-1j * st, ct]], dtype=np.complex128)


def ZZ_phase(theta: float) -> np.ndarray:
    # exp(+i theta Z⊗Z) -> diag(e^{iθ}, e^{-iθ}, e^{-iθ}, e^{iθ})
    p = np.exp(1j * theta)
    m = np.exp(-1j * theta)
    return np.diag([p, m, m, p]).astype(np.complex128)


def qaoa_state(n: int, edges, w: np.ndarray, params: np.ndarray, p: int) -> np.ndarray:
    """
    params = [γ_1..γ_p, β_1..β_p]
    Up to global phase, cost unitary:
      U_C = Π_e exp(+i (γ w_e / 2) Z_i Z_j).
    """
    gammas = params[:p]
    betas = params[p:2 * p]

    K = 1 << n
    psi = np.ones(K, dtype=np.complex128) / math.sqrt(K)  # |+>^n

    for l in range(p):
        gamma = float(gammas[l])
        beta = float(betas[l])

        # cost layer
        for (i, j), wij in zip(edges, w):
            theta = 0.5 * gamma * float(wij)
            psi = _apply_2q(psi, ZZ_phase(theta), int(i), int(j), n)

        # mixer layer
        gate = RX(2.0 * beta)  # exp(-i beta X) = Rx(2 beta)
        for q in range(n):
            psi = _apply_1q(psi, gate, q, n)

        psi = _renorm(psi)

    return psi


def qaoa_energy(n: int, edges, cut_mask: np.ndarray, w: np.ndarray,
                params: np.ndarray, p: int) -> float:
    psi = qaoa_state(n, edges, w, params, p)
    J, _, _ = expect_J(psi, cut_mask, w)
    return -J


# ==============================================================================
# 8) Inner optimizer: SPSA (cost model as in main scripts)
# ==============================================================================

def spsa_minimize(energy_fun, p0: np.ndarray, bounds, iters: int, seed: int,
                  a: float = 0.2, c: float = 0.12, A: float = 20.0,
                  alpha: float = 0.602, gamma: float = 0.101):
    rng = np.random.default_rng(to_uint_seed(seed))
    p = p0.astype(float).copy()
    lo = np.array([b[0] for b in bounds], float)
    hi = np.array([b[1] for b in bounds], float)
    best_p, best_E = p.copy(), float("inf")
    evals = 0
    for k in range(1, iters + 1):
        ak = a / ((k + A) ** alpha)
        ck = c / (k ** gamma)
        delta = rng.choice([-1.0, 1.0], size=p.size)
        Ep = float(energy_fun(np.clip(p + ck * delta, lo, hi)))
        Em = float(energy_fun(np.clip(p - ck * delta, lo, hi)))
        evals += 2
        ghat = (Ep - Em) / (2.0 * ck) * delta
        p = np.clip(p - ak * ghat, lo, hi)
        E = float(energy_fun(p))
        evals += 1
        if E < best_E:
            best_E, best_p = E, p.copy()
    return best_p, best_E, evals


# ==============================================================================
# 9) Classical envelope maximum for normalization J*
# ==============================================================================

def classical_Jstar(fam: Family1D, cut_mask: np.ndarray, grid_points: int) -> Tuple[float, float]:
    """
    Compute J* = max_{λ,z} J(z;λ) approximately on a λ grid.
    Returns (J_star, lam_star_grid).
    """
    lams = np.linspace(fam.lam_min, fam.lam_max, int(grid_points))
    bestJ = -1e30
    bestLam = float(lams[0])
    for lam in lams:
        w = fam.w(float(lam)).astype(np.float64)
        cut_vals = cut_mask @ w  # shape (K,)
        j = float(np.max(cut_vals))
        if j > bestJ:
            bestJ = j
            bestLam = float(lam)
    if not np.isfinite(bestJ) or bestJ <= 0:
        bestJ = 1.0
    return float(bestJ), float(bestLam)


# ==============================================================================
# 10) Readout metrics
# ==============================================================================

def sample_indices(rng: np.random.Generator, probs: np.ndarray, shots: int) -> np.ndarray:
    shots = int(shots)
    if shots <= 0:
        return np.empty(0, dtype=np.int64)
    return rng.choice(probs.size, size=shots, replace=True, p=probs).astype(np.int64)


def best_of_prefix(idx: np.ndarray, cut_vals: np.ndarray, shots: int) -> float:
    shots = int(shots)
    if shots <= 0:
        return float("nan")
    if idx.size < shots:
        raise ValueError("idx must have at least 'shots' entries.")
    return float(np.max(cut_vals[idx[:shots]]))


def tail_probability(probs: np.ndarray,
                     cut_vals: np.ndarray,
                     *,
                     metric: str,
                     J_star: float,
                     hit_eps: float,
                     topk_frac: float) -> Tuple[float, float]:
    """
    Returns (p_tail, threshold) where
      - metric == 'hit' : threshold = (1-hit_eps)*J_star
      - metric == 'topk': threshold = quantile_{1-topk_frac}(cut_vals)
    and p_tail = P(cut >= threshold) under 'probs'.

    This is computed EXACTLY from the state distribution (no shot noise).
    """
    metric = str(metric).lower().strip()
    if metric == "hit":
        thr = (1.0 - float(hit_eps)) * float(J_star)
    elif metric == "topk":
        q = 1.0 - float(topk_frac)
        q = float(np.clip(q, 0.0, 1.0))
        thr = float(np.quantile(cut_vals, q))
    else:
        return float("nan"), float("nan")

    m = cut_vals >= thr
    p = float(np.sum(probs[m]))
    p = float(np.clip(p, 0.0, 1.0))
    return p, float(thr)


# ==============================================================================
# 11) Outer loops
# ==============================================================================

def run_outer_vqe_id(n: int,
                     cut_mask: np.ndarray,
                     fam: Family1D,
                     *,
                     J_star: float,
                     lam0: float,
                     outer: int,
                     inner: int,
                     eta0: float,
                     eta_pow: float,
                     step_clip: float,
                     seed: int,
                     L_vqe: int,
                     readout_shots: int,
                     shot_list: List[int],
                     readout_seed: int,
                     tail_metric: str,
                     hit_eps: float,
                     topk_frac: float):
    """
    VQE outer optimization using exact reuse hypergradient:
      g = dw/dλ · p_cut

    Additionally tracks:
      - best-of-S readout for S in shot_list
      - optional tail probability metric from the exact state distribution
    """
    lam_min, lam_max = fam.lam_min, fam.lam_max
    lam = float(np.clip(lam0, lam_min, lam_max))

    D = 2 * n * L_vqe
    params = np.zeros(D, float)
    bounds = [(-math.pi, math.pi)] * D

    shot_list = sorted({int(s) for s in shot_list if (int(s) > 0 and int(s) <= int(readout_shots))})
    if int(readout_shots) not in shot_list:
        shot_list.append(int(readout_shots))
    shot_list = sorted(set(shot_list))
    S_max = int(readout_shots)  # fixed per-step budget; smaller S use prefixes

    rng_read = np.random.default_rng(to_uint_seed(readout_seed))

    hist = {
        "lam": [], "J": [], "J_best": [], "evals_cum": [],
        "ro_best": [], "ro_best_sofar": [],
        "tail_prob": [], "tail_best_sofar": [],
    }
    # per-S best-so-far
    bestR = {S: -1e18 for S in shot_list}

    evals = 0.0
    bestJ = -1e18
    bestTail = -1.0

    for t in range(1, int(outer) + 1):
        w = fam.w(lam)

        def Efun(pvec): return vqe_energy(n, cut_mask, w, pvec, L_vqe)

        params, _, ev_in = spsa_minimize(Efun, params, bounds, iters=inner, seed=seed + 1000 * t)
        evals += float(ev_in)

        # expectation at λ
        psi = vqe_state(n, params, L_vqe)
        J, p_cut, probs = expect_J(psi, cut_mask, w)
        evals += 1.0
        bestJ = max(bestJ, float(J))

        # precompute per-bitstring cut values for readout metrics
        cut_vals = (cut_mask @ w).astype(np.float64)  # shape (K,)

        # readout sample once at max shots, then reuse prefixes for smaller S
        idx = sample_indices(rng_read, probs, S_max)

        for S in shot_list:
            rb = best_of_prefix(idx, cut_vals, S)
            bestR[S] = max(bestR[S], float(rb))
            # store per-S running best
            key = f"ro_best_sofar_S{S}"
            if key not in hist:
                hist[key] = []
            hist[key].append(float(bestR[S]))

        # main readout metric (S = readout_shots)
        rb_main = best_of_prefix(idx, cut_vals, int(readout_shots))
        hist["ro_best"].append(float(rb_main))
        hist["ro_best_sofar"].append(float(bestR[int(readout_shots)]))

        # tail probability (exact, no shot noise)
        if str(tail_metric).lower() != "none":
            p_tail, _thr = tail_probability(
                probs, cut_vals,
                metric=tail_metric,
                J_star=J_star,
                hit_eps=hit_eps,
                topk_frac=topk_frac
            )
            bestTail = max(bestTail, float(p_tail))
            hist["tail_prob"].append(float(p_tail))
            hist["tail_best_sofar"].append(float(bestTail))
        else:
            hist["tail_prob"].append(float("nan"))
            hist["tail_best_sofar"].append(float("nan"))

        # hypergradient (reuse term; exact for VQE)
        g = float(fam.dw_dlam(lam) @ p_cut)

        # outer step
        eta = eta0 / (t ** eta_pow)
        step = float(np.clip(eta * g, -step_clip, step_clip))
        lam = float(np.clip(lam + step, lam_min, lam_max))

        hist["lam"].append(lam)
        hist["J"].append(float(J))
        hist["J_best"].append(float(bestJ))
        hist["evals_cum"].append(float(evals))

    for k in hist:
        hist[k] = np.asarray(hist[k], float)
    return hist


def run_outer_qaoa_full(n: int,
                       edges,
                       cut_mask: np.ndarray,
                       fam: Family1D,
                       *,
                       J_star: float,
                       lam0: float,
                       outer: int,
                       inner: int,
                       eta0: float,
                       eta_pow: float,
                       step_clip: float,
                       seed: int,
                       p_qaoa: int,
                       fd_c_frac: float,
                       readout_shots: int,
                       shot_list: List[int],
                       readout_seed: int,
                       tail_metric: str,
                       hit_eps: float,
                       topk_frac: float):
    """
    QAOA outer optimization using fixed-θ FD hypergradient:
      g_full ≈ [J(θ,λ+c) - J(θ,λ-c)] / (2c)
    which includes explicit + state-dependence terms.

    Additionally tracks:
      - best-of-S readout for S in shot_list
      - optional tail probability metric from the exact state distribution
    """
    lam_min, lam_max = fam.lam_min, fam.lam_max
    lam = float(np.clip(lam0, lam_min, lam_max))

    D = 2 * p_qaoa
    params = np.zeros(D, float)
    bounds = [(-math.pi, math.pi)] * D

    shot_list = sorted({int(s) for s in shot_list if (int(s) > 0 and int(s) <= int(readout_shots))})
    if int(readout_shots) not in shot_list:
        shot_list.append(int(readout_shots))
    shot_list = sorted(set(shot_list))
    S_max = int(readout_shots)  # fixed per-step budget; smaller S use prefixes

    rng_read = np.random.default_rng(to_uint_seed(readout_seed))

    c_fd = float(fd_c_frac * (lam_max - lam_min))

    hist = {
        "lam": [], "J": [], "J_best": [], "evals_cum": [],
        "ro_best": [], "ro_best_sofar": [],
        "tail_prob": [], "tail_best_sofar": [],
    }
    bestR = {S: -1e18 for S in shot_list}

    evals = 0.0
    bestJ = -1e18
    bestTail = -1.0

    for t in range(1, int(outer) + 1):
        w = fam.w(lam)

        def Efun(pvec): return qaoa_energy(n, edges, cut_mask, w, pvec, p_qaoa)

        params, _, ev_in = spsa_minimize(Efun, params, bounds, iters=inner, seed=seed + 1000 * t)
        evals += float(ev_in)

        # expectation at λ
        psi = qaoa_state(n, edges, w, params, p_qaoa)
        J, _p_cut, probs = expect_J(psi, cut_mask, w)
        evals += 1.0
        bestJ = max(bestJ, float(J))

        cut_vals = (cut_mask @ w).astype(np.float64)

        # readout (sample once, reuse prefixes)
        idx = sample_indices(rng_read, probs, S_max)

        for S in shot_list:
            rb = best_of_prefix(idx, cut_vals, S)
            bestR[S] = max(bestR[S], float(rb))
            key = f"ro_best_sofar_S{S}"
            if key not in hist:
                hist[key] = []
            hist[key].append(float(bestR[S]))

        rb_main = best_of_prefix(idx, cut_vals, int(readout_shots))
        hist["ro_best"].append(float(rb_main))
        hist["ro_best_sofar"].append(float(bestR[int(readout_shots)]))

        # tail probability (exact)
        if str(tail_metric).lower() != "none":
            p_tail, _thr = tail_probability(
                probs, cut_vals,
                metric=tail_metric,
                J_star=J_star,
                hit_eps=hit_eps,
                topk_frac=topk_frac
            )
            bestTail = max(bestTail, float(p_tail))
            hist["tail_prob"].append(float(p_tail))
            hist["tail_best_sofar"].append(float(bestTail))
        else:
            hist["tail_prob"].append(float("nan"))
            hist["tail_best_sofar"].append(float("nan"))

        # fixed-θ FD hypergradient (2 extra objective evals)
        lp = float(np.clip(lam + c_fd, lam_min, lam_max))
        lm = float(np.clip(lam - c_fd, lam_min, lam_max))
        if abs(lp - lm) < 1e-12:
            g = 0.0
        else:
            wp = fam.w(lp)
            wm = fam.w(lm)

            psi_p = qaoa_state(n, edges, wp, params, p_qaoa)
            Jp, _, _ = expect_J(psi_p, cut_mask, wp)

            psi_m = qaoa_state(n, edges, wm, params, p_qaoa)
            Jm, _, _ = expect_J(psi_m, cut_mask, wm)

            evals += 2.0
            g = float((Jp - Jm) / (lp - lm))

        # outer step
        eta = eta0 / (t ** eta_pow)
        step = float(np.clip(eta * g, -step_clip, step_clip))
        lam = float(np.clip(lam + step, lam_min, lam_max))

        hist["lam"].append(lam)
        hist["J"].append(float(J))
        hist["J_best"].append(float(bestJ))
        hist["evals_cum"].append(float(evals))

    for k in hist:
        hist[k] = np.asarray(hist[k], float)
    return hist


# ==============================================================================
# 12) Plotting helpers
# ==============================================================================

def _mean_stderr(Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.mean(Y, axis=0)
    if Y.shape[0] > 1:
        se = np.std(Y, axis=0, ddof=1) / math.sqrt(Y.shape[0])
    else:
        se = np.zeros_like(mu)
    return mu, se


def plot_expect_and_readout_vs_evals(path: Path,
                                     curves_expect: Dict[str, List[Dict]],
                                     curves_readout: Dict[str, List[Dict]],
                                     budget: float,
                                     readout_shots: int,
                                     annotate: bool = False):
    """
    Side-by-side (1x2) main figure:
      left  : best-so-far expectation (J/J*)
      right : best-so-far sampled cut (best-of-S readout, /J*)
    No panel labels (a/b). No annotation text boxes. Legend only once.
    """
    set_pub_style(grid=False)

    # two-column width for side-by-side
    FULL_W = 6.95   # inches
    H = H_COL       # keep your single-panel height

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(FULL_W, H), constrained_layout=True
    )

    E = np.linspace(0.0, float(budget), 320)

    # expectation stats
    stats_exp = {}
    for name, runs in curves_expect.items():
        Y = np.vstack([step_sample(r["evals"], r["best"], E) for r in runs])
        stats_exp[name] = _mean_stderr(Y)

    # readout stats
    stats_ro = {}
    for name, runs in curves_readout.items():
        Y = np.vstack([step_sample(r["evals"], r["best"], E) for r in runs])
        stats_ro[name] = _mean_stderr(Y)

    def draw(ax, stats, name, color, ls, label, alpha_fill, z):
        mu, se = stats[name]
        ax.plot(E, mu, color=color, lw=1.8, ls=ls, label=label, zorder=z)
        ax.fill_between(E, mu - se, mu + se, color=color, alpha=alpha_fill, lw=0, zorder=z - 1)

    # Left: expectation
    draw(ax1, stats_exp, "VQE",  COLORS["VQE"],  "-",  "_nolegend_", 0.12, z=3)
    draw(ax1, stats_exp, "QAOA", COLORS["QAOA"], "--", "_nolegend_", 0.10, z=4)
    ax1.set_ylabel(r"Best-so-far expectation $J/J^*$")

    # Right: readout
    draw(ax2, stats_ro, "VQE",  COLORS["VQE"],  "-",  "ID(VQE)", 0.12, z=3)
    draw(ax2, stats_ro, "QAOA", COLORS["QAOA"], "--", r"ID(QAOA, full $\partial_\lambda J$)", 0.10, z=4)
    ax2.set_ylabel(r"Best sampled cut $/J^*$")
    ax2.set_xlabel("Energy evaluations")

    # Limits
    for ax, stats in [(ax1, stats_exp), (ax2, stats_ro)]:
        ax.set_xlim(0.0, float(budget))
        mus = [v[0] for v in stats.values()]
        y_max = float(np.max(np.vstack(mus))) if mus else 1.0
        ax.set_ylim(0.0, min(1.25, max(0.65, y_max * 1.10)))

    # Legend only once (right panel)
    leg = ax2.legend(loc="lower right", frameon=True, fancybox=False, framealpha=0.88)
    leg.get_frame().set_linewidth(0.0)
    leg.get_frame().set_facecolor("white")

    _savefig(fig, path)


def plot_tailprob_vs_evals(path: Path,
                           curves_tail: Dict[str, List[Dict]],
                           budget: float,
                           tail_label: str,
                           annotate: bool = False):
    """
    Optional: Tail probability (best-so-far) vs energy-eval budget.
    (No in-plot annotation boxes; legend only.)
    """
    set_pub_style(grid=False)
    fig, ax = plt.subplots(figsize=fig_size(), constrained_layout=True)

    E = np.linspace(0.0, float(budget), 320)

    stats = {}
    for name, runs in curves_tail.items():
        Y = np.vstack([step_sample(r["evals"], r["best"], E) for r in runs])
        stats[name] = _mean_stderr(Y)

    def draw(name: str, color: str, ls: str, label: str, alpha_fill: float, z: int):
        mu, se = stats[name]
        ax.plot(E, mu, color=color, lw=1.8, ls=ls, label=label, zorder=z)
        ax.fill_between(E, mu - se, mu + se, color=color, alpha=alpha_fill, lw=0, zorder=z - 1)

    draw("VQE", COLORS["VQE"], "-", "ID(VQE)", 0.12, z=3)
    draw("QAOA", COLORS["QAOA"], "--", r"ID(QAOA, full $\partial_\lambda J$)", 0.10, z=4)

    ax.set_xlabel("Energy evaluations")
    ax.set_ylabel(tail_label)
    ax.set_xlim(0.0, float(budget))
    ax.set_ylim(0.0, 1.02)

    leg = ax.legend(loc="lower right", frameon=True, fancybox=False, framealpha=0.88)
    leg.get_frame().set_linewidth(0.0)
    leg.get_frame().set_facecolor("white")

    _savefig(fig, path)


def plot_evals_vs_outer(path: Path,
                        evals_mean: Dict[str, np.ndarray]):
    """Optional: mean cumulative energy evals vs outer iteration."""
    set_pub_style(grid=False)
    fig, ax = plt.subplots(figsize=fig_size(), constrained_layout=True)

    for name, ev_mean in evals_mean.items():
        t = np.arange(1, ev_mean.size + 1)
        if name == "VQE":
            ax.plot(t, ev_mean, color=COLORS["VQE"], lw=1.8, label="ID(VQE)", zorder=3)
        elif name == "QAOA":
            ax.plot(t, ev_mean, color=COLORS["QAOA"], lw=1.8, ls="--",
                    label=r"ID(QAOA, full $\partial_\lambda J$)", zorder=4)
        else:
            ax.plot(t, ev_mean, lw=1.5, label=name)

    ax.set_xlabel(r"Outer iteration $t$")
    ax.set_ylabel("Cumulative energy evaluations")

    leg = ax.legend(loc="upper left", frameon=True, fancybox=False, framealpha=0.88)
    leg.get_frame().set_linewidth(0.0)
    leg.get_frame().set_facecolor("white")

    _savefig(fig, path)


def plot_tradeoff_scatter(path: Path,
                          x_vqe: np.ndarray, y_vqe: np.ndarray,
                          x_qaoa: np.ndarray, y_qaoa: np.ndarray,
                          xlabel: str, ylabel: str,
                          annotate: bool = False):
    """
    Optional: Scatter at budget B.
      x = expectation metric, y = readout metric.
      Diagonal y=x is a reference: points above it have stronger tail advantage.

    (No in-plot annotation boxes; legend only.)
    """
    set_pub_style(grid=False)
    fig, ax = plt.subplots(figsize=fig_size(), constrained_layout=True)

    x_vqe = np.asarray(x_vqe, float)
    y_vqe = np.asarray(y_vqe, float)
    x_qaoa = np.asarray(x_qaoa, float)
    y_qaoa = np.asarray(y_qaoa, float)

    ax.scatter(x_vqe, y_vqe, s=26, alpha=0.9,
               color=COLORS["VQE"], edgecolors="white", linewidths=0.3,
               label="ID(VQE)", zorder=3)
    ax.scatter(x_qaoa, y_qaoa, s=26, alpha=0.9,
               color=COLORS["QAOA"], edgecolors="white", linewidths=0.3,
               label=r"ID(QAOA, full $\partial_\lambda J$)", zorder=4)

    # y=x diagonal
    lo = float(min(np.min(x_vqe), np.min(y_vqe), np.min(x_qaoa), np.min(y_qaoa)))
    hi = float(max(np.max(x_vqe), np.max(y_vqe), np.max(x_qaoa), np.max(y_qaoa)))
    pad = 0.05 * (hi - lo + 1e-12)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
            color="black", lw=0.9, ls=":", zorder=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_xlim(max(0.0, lo - pad), min(1.02, hi + pad))
    ax.set_ylim(max(0.0, lo - pad), min(1.02, hi + pad))

    leg = ax.legend(loc="lower right", frameon=True, fancybox=False, framealpha=0.88)
    leg.get_frame().set_linewidth(0.0)
    leg.get_frame().set_facecolor("white")

    _savefig(fig, path)


def plot_readout_shots_sweep(path: Path,
                             shots: List[int],
                             stats_vqe: List[Tuple[float, float]],
                             stats_qaoa: List[Tuple[float, float]],
                             budget: float,
                             annotate: bool = False):
    """
    Optional: At fixed eval budget B, show mean±stderr of best-of-S readout vs S.
    (No in-plot annotation boxes; legend only.)
    """
    set_pub_style(grid=False)
    fig, ax = plt.subplots(figsize=fig_size(), constrained_layout=True)

    shots = [int(s) for s in shots]
    x = np.asarray(shots, float)

    vqe_mu = np.asarray([m for (m, _se) in stats_vqe], float)
    vqe_se = np.asarray([_se for (_m, _se) in stats_vqe], float)

    qaoa_mu = np.asarray([m for (m, _se) in stats_qaoa], float)
    qaoa_se = np.asarray([_se for (_m, _se) in stats_qaoa], float)

    ax.errorbar(x, vqe_mu, yerr=vqe_se, color=COLORS["VQE"], lw=1.6, ls="-",
                marker="o", ms=3.5, capsize=2.5, label="ID(VQE)", zorder=3)
    ax.errorbar(x, qaoa_mu, yerr=qaoa_se, color=COLORS["QAOA"], lw=1.6, ls="--",
                marker="s", ms=3.3, capsize=2.5, label=r"ID(QAOA, full $\partial_\lambda J$)", zorder=4)

    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlabel(r"Readout shots per outer step $S$")
    ax.set_ylabel(r"Best-of-$S$ at budget $B$  ($/J^*$)")
    ax.set_ylim(0.0, 1.02)

    leg = ax.legend(loc="lower right", frameon=True, fancybox=False, framealpha=0.88)
    leg.get_frame().set_linewidth(0.0)
    leg.get_frame().set_facecolor("white")

    _savefig(fig, path)


# ==============================================================================
# 13) Main
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="out_exp2C_readout_bridge")
    p.add_argument("--fmt", type=str, default="pdf", choices=["pdf", "png", "svg"])

    # instance set
    p.add_argument("--seeds", type=str, default="7,8")
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--p_edge", type=float, default=0.45)

    # family
    p.add_argument("--kind", type=str, default="quadratic", choices=["linear", "quadratic", "periodic"])
    p.add_argument("--periodic_K", type=int, default=6)
    p.add_argument("--lam_min", type=float, default=-5.0)
    p.add_argument("--lam_max", type=float, default=5.0)
    p.add_argument("--lam0", type=float, default=4)
    p.add_argument("--grid", type=int, default=401)

    # ansatz depths
    p.add_argument("--L_vqe", type=int, default=2)
    p.add_argument("--p_qaoa", type=int, default=3)

    # budgets
    p.add_argument("--outer", type=int, default=10)
    p.add_argument("--inner", type=int, default=100)

    # outer schedule
    p.add_argument("--eta0", type=float, default=0.25)
    p.add_argument("--eta_pow", type=float, default=0.4)
    p.add_argument("--step_clip", type=float, default=0.6)

    # QAOA FD step
    p.add_argument("--fd_c_frac", type=float, default=0.10)

    # readout realism
    p.add_argument("--readout_shots", type=int, default=128,
                   help="Main readout budget S per outer step for the main 2-panel figure.")
    p.add_argument("--supp_shots", type=str, default="2,4,8,16,32,64,128",
                   help="Comma-separated additional shot budgets for a sweep plot (supplement). "
                        "Set to '' to disable.")

    # tail metric (optional, recommended)
    p.add_argument("--tail_metric", type=str, default="hit", choices=["none", "hit", "topk"],
                   help="Tail metric to plot: "
                        "'hit' = P(cut >= (1-eps)J*), "
                        "'topk' = P(cut in top-k%% bitstrings).")
    p.add_argument("--hit_eps", type=float, default=0.10,
                   help="ε for hit-rate threshold (1-ε)J*. Default 0.10 => 90%% of J*.")
    p.add_argument("--topk_frac", type=float, default=0.01,
                   help="k for top-k tail probability (fraction, not percent). Default 0.01 => top 1%%.")

    # plots toggles (default ON for paper convenience)
    p.set_defaults(cost_plot=True)
    p.add_argument("--no_cost_plot", action="store_false", dest="cost_plot",
                   help="Disable the evals-vs-outer cost plot.")
    p.set_defaults(tradeoff_plot=True)
    p.add_argument("--no_tradeoff_plot", action="store_false", dest="tradeoff_plot",
                   help="Disable the tradeoff scatter plot at budget B.")
    p.set_defaults(shots_sweep_plot=True)
    p.add_argument("--no_shots_sweep_plot", action="store_false", dest="shots_sweep_plot",
                   help="Disable the readout-shot sweep plot (if supp_shots provided).")

    # kept for compatibility; plots ignore annotate anyway (legend-only policy)
    p.add_argument("--no_annotate", action="store_true")

    return p.parse_args()


def main():
    a = parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    seeds = parse_int_list(a.seeds)
    if not seeds:
        raise ValueError("No seeds provided.")

    # readout shot list for sweep (includes main S)
    supp_raw = parse_int_list(a.supp_shots)
    # Supplement is meant for *smaller* shot budgets (e.g. 16/32/64) under the SAME per-step budget S.
    supp = [int(x) for x in supp_raw if (int(x) > 0 and int(x) <= int(a.readout_shots))]
    ignored = [int(x) for x in supp_raw if int(x) > int(a.readout_shots)]
    if ignored:
        print(f"[warn] ignoring supp_shots > readout_shots (per-step budget): {ignored}")
    shot_list = sorted(set([int(a.readout_shots)] + supp))
    if not shot_list:
        shot_list = [int(a.readout_shots)]

    Z = precompute_z_big_endian(int(a.n))

    # step curves (budget-first)
    curves_expect = {"VQE": [], "QAOA": []}
    curves_readout_main = {"VQE": [], "QAOA": []}
    curves_tail = {"VQE": [], "QAOA": []}

    # per-shot sweep curves (for budget-B extraction)
    curves_readout_byS: Dict[int, Dict[str, List[Dict]]] = {S: {"VQE": [], "QAOA": []} for S in shot_list}

    # store evals curves for cost plot
    evals_curves = {"VQE": [], "QAOA": []}

    rows = []

    for s in seeds:
        rng = np.random.default_rng(to_uint_seed(s))
        edges = generate_random_graph(int(a.n), float(a.p_edge), rng)
        retry = 0
        while (not edges) and retry < 50:
            retry += 1
            edges = generate_random_graph(int(a.n), float(a.p_edge), rng)
        if not edges:
            print(f"[warn] skip seed={s} (empty graph)")
            continue

        cut_mask = build_cut_mask(edges, Z)
        fam = Family1D(len(edges), a.kind, (a.lam_min, a.lam_max), rng, K=int(a.periodic_K))
        J_star, lam_star = classical_Jstar(fam, cut_mask, int(a.grid))

        hist_vqe = run_outer_vqe_id(
            a.n, cut_mask, fam,
            J_star=J_star,
            lam0=a.lam0, outer=a.outer, inner=a.inner,
            eta0=a.eta0, eta_pow=a.eta_pow, step_clip=a.step_clip,
            seed=s + 0,
            L_vqe=a.L_vqe,
            readout_shots=a.readout_shots,
            shot_list=shot_list,
            readout_seed=s + 4242,
            tail_metric=a.tail_metric,
            hit_eps=a.hit_eps,
            topk_frac=a.topk_frac
        )

        hist_qaoa = run_outer_qaoa_full(
            a.n, edges, cut_mask, fam,
            J_star=J_star,
            lam0=a.lam0, outer=a.outer, inner=a.inner,
            eta0=a.eta0, eta_pow=a.eta_pow, step_clip=a.step_clip,
            seed=s + 100000,
            p_qaoa=a.p_qaoa,
            fd_c_frac=a.fd_c_frac,
            readout_shots=a.readout_shots,
            shot_list=shot_list,
            readout_seed=s + 8888,
            tail_metric=a.tail_metric,
            hit_eps=a.hit_eps,
            topk_frac=a.topk_frac
        )

        # normalize curves
        vqe_exp = np.clip(hist_vqe["J_best"] / J_star, 0.0, 1.5)
        qaoa_exp = np.clip(hist_qaoa["J_best"] / J_star, 0.0, 1.5)

        vqe_ro_main = np.clip(hist_vqe[f"ro_best_sofar_S{int(a.readout_shots)}"] / J_star, 0.0, 1.5)
        qaoa_ro_main = np.clip(hist_qaoa[f"ro_best_sofar_S{int(a.readout_shots)}"] / J_star, 0.0, 1.5)

        curves_expect["VQE"].append({"evals": hist_vqe["evals_cum"], "best": vqe_exp})
        curves_expect["QAOA"].append({"evals": hist_qaoa["evals_cum"], "best": qaoa_exp})

        curves_readout_main["VQE"].append({"evals": hist_vqe["evals_cum"], "best": vqe_ro_main})
        curves_readout_main["QAOA"].append({"evals": hist_qaoa["evals_cum"], "best": qaoa_ro_main})

        # per-shot curves (for sweep)
        for S in shot_list:
            vqe_ro_S = np.clip(hist_vqe[f"ro_best_sofar_S{S}"] / J_star, 0.0, 1.5)
            qaoa_ro_S = np.clip(hist_qaoa[f"ro_best_sofar_S{S}"] / J_star, 0.0, 1.5)
            curves_readout_byS[S]["VQE"].append({"evals": hist_vqe["evals_cum"], "best": vqe_ro_S})
            curves_readout_byS[S]["QAOA"].append({"evals": hist_qaoa["evals_cum"], "best": qaoa_ro_S})

        # tail curves
        if str(a.tail_metric).lower() != "none":
            vqe_tail = np.clip(hist_vqe["tail_best_sofar"], 0.0, 1.0)
            qaoa_tail = np.clip(hist_qaoa["tail_best_sofar"], 0.0, 1.0)
            curves_tail["VQE"].append({"evals": hist_vqe["evals_cum"], "best": vqe_tail})
            curves_tail["QAOA"].append({"evals": hist_qaoa["evals_cum"], "best": qaoa_tail})

        # cost curves
        evals_curves["VQE"].append(hist_vqe["evals_cum"])
        evals_curves["QAOA"].append(hist_qaoa["evals_cum"])

        # per-seed row (final-at-end, plus settings)
        row = {
            "seed": int(s),
            "kind": a.kind,
            "n": int(a.n),
            "m_edges": int(len(edges)),
            "J_star_grid": float(J_star),
            "lam_star_grid": float(lam_star),
            "outer": int(a.outer),
            "inner": int(a.inner),
            "L_vqe": int(a.L_vqe),
            "p_qaoa": int(a.p_qaoa),
            "eta0": float(a.eta0),
            "eta_pow": float(a.eta_pow),
            "step_clip": float(a.step_clip),
            "fd_c_frac": float(a.fd_c_frac),
            "readout_shots": int(a.readout_shots),
            "supp_shots": a.supp_shots,
            "tail_metric": str(a.tail_metric),
            "hit_eps": float(a.hit_eps),
            "topk_frac": float(a.topk_frac),
            "vqe_exp_final": float(vqe_exp[-1]),
            "qaoa_exp_final": float(qaoa_exp[-1]),
            "vqe_ro_final": float(vqe_ro_main[-1]),
            "qaoa_ro_final": float(qaoa_ro_main[-1]),
            "vqe_evals_final": float(hist_vqe["evals_cum"][-1]),
            "qaoa_evals_final": float(hist_qaoa["evals_cum"][-1]),
        }
        if str(a.tail_metric).lower() != "none":
            row.update({
                "vqe_tail_final": float(vqe_tail[-1]),
                "qaoa_tail_final": float(qaoa_tail[-1]),
            })
        rows.append(row)

        msg = (f"[seed={s}] exp_final: VQE={vqe_exp[-1]:.3f} | QAOA={qaoa_exp[-1]:.3f} || "
               f"readout_final(S={a.readout_shots}): VQE={vqe_ro_main[-1]:.3f} | QAOA={qaoa_ro_main[-1]:.3f}")
        if str(a.tail_metric).lower() != "none":
            msg += f" || tail_final: VQE={vqe_tail[-1]:.3f} | QAOA={qaoa_tail[-1]:.3f}"
        print(msg)

    if not rows:
        raise RuntimeError("No runs completed.")

    # common energy-eval budget B for fair curves
    all_end = []
    for name in ["VQE", "QAOA"]:
        for r in curves_expect[name]:
            all_end.append(float(r["evals"][-1]))
    B = float(min(all_end))

    # mean evals-vs-outer (cost plot)
    evals_mean = {}
    for name, arrs in evals_curves.items():
        if not arrs:
            continue
        M = np.vstack(arrs)  # (N, outer)
        evals_mean[name] = np.mean(M, axis=0)

    # helpers
    def budget_stats(curves: Dict[str, List[Dict]], name: str) -> Tuple[float, float]:
        vals = [step_value_at(r["evals"], r["best"], B) for r in curves[name]]
        vals = np.asarray(vals, float)
        mu = float(np.mean(vals))
        se = float(np.std(vals, ddof=1) / math.sqrt(vals.size)) if vals.size > 1 else 0.0
        return mu, se

    def auc_stats(curves: Dict[str, List[Dict]], name: str) -> Tuple[float, float]:
        vals = [step_auc(r["evals"], r["best"], B) / (B * 1.0) for r in curves[name]]
        vals = np.asarray(vals, float)
        mu = float(np.mean(vals))
        se = float(np.std(vals, ddof=1) / math.sqrt(vals.size)) if vals.size > 1 else 0.0
        return mu, se

    # expectation + readout stats @B
    vqe_exp_mu, vqe_exp_se = budget_stats(curves_expect, "VQE")
    qaoa_exp_mu, qaoa_exp_se = budget_stats(curves_expect, "QAOA")
    vqe_ro_mu, vqe_ro_se = budget_stats(curves_readout_main, "VQE")
    qaoa_ro_mu, qaoa_ro_se = budget_stats(curves_readout_main, "QAOA")

    vqe_exp_auc, vqe_exp_auc_se = auc_stats(curves_expect, "VQE")
    qaoa_exp_auc, qaoa_exp_auc_se = auc_stats(curves_expect, "QAOA")
    vqe_ro_auc, vqe_ro_auc_se = auc_stats(curves_readout_main, "VQE")
    qaoa_ro_auc, qaoa_ro_auc_se = auc_stats(curves_readout_main, "QAOA")

    # gap / closure summaries
    gap_exp = vqe_exp_mu - qaoa_exp_mu
    gap_ro = vqe_ro_mu - qaoa_ro_mu
    gap_closure = float("nan")
    if abs(gap_exp) > 1e-12 and np.isfinite(gap_exp) and np.isfinite(gap_ro):
        gap_closure = 1.0 - (gap_ro / gap_exp)

    # tail factor: how much sampling boosts over expectation
    tail_factor_vqe = float(vqe_ro_mu / max(vqe_exp_mu, 1e-12))
    tail_factor_qaoa = float(qaoa_ro_mu / max(qaoa_exp_mu, 1e-12))

    # tail metric stats (optional)
    tail_mu = tail_se = tail_auc_mu = tail_auc_se = None
    if str(a.tail_metric).lower() != "none" and curves_tail["VQE"]:
        vqe_tail_mu, vqe_tail_se = budget_stats(curves_tail, "VQE")
        qaoa_tail_mu, qaoa_tail_se = budget_stats(curves_tail, "QAOA")
        vqe_tail_auc, vqe_tail_auc_se = auc_stats(curves_tail, "VQE")
        qaoa_tail_auc, qaoa_tail_auc_se = auc_stats(curves_tail, "QAOA")
        tail_mu = (vqe_tail_mu, qaoa_tail_mu)
        tail_se = (vqe_tail_se, qaoa_tail_se)
        tail_auc_mu = (vqe_tail_auc, qaoa_tail_auc)
        tail_auc_se = (vqe_tail_auc_se, qaoa_tail_auc_se)

    # shot sweep stats (optional)
    sweep_stats = []
    for S in shot_list:
        mu_v, se_v = budget_stats(curves_readout_byS[S], "VQE")
        mu_q, se_q = budget_stats(curves_readout_byS[S], "QAOA")
        sweep_stats.append((S, mu_v, se_v, mu_q, se_q))

    # write CSV + summary
    write_csv(out / "exp2C_results.csv", rows)

    lines = []
    lines.append(f"Experiment 2C (upgraded) | kind={a.kind} | seeds={len(rows)} | budget B={B:.1f} evals")
    lines.append(f"Settings: eta0={a.eta0} eta_pow={a.eta_pow} step_clip={a.step_clip} "
                 f"fd_c_frac={a.fd_c_frac} | readout_shots={a.readout_shots} | supp_shots={a.supp_shots}")
    lines.append("")

    lines.append("Best-so-far ratios at budget B (mean ± stderr):")
    lines.append(f"  Expectation  ID(VQE):   {vqe_exp_mu:.4f} ± {vqe_exp_se:.4f}")
    lines.append(f"  Expectation  ID(QAOA):  {qaoa_exp_mu:.4f} ± {qaoa_exp_se:.4f}")
    lines.append(f"  Readout best ID(VQE):   {vqe_ro_mu:.4f} ± {vqe_ro_se:.4f}")
    lines.append(f"  Readout best ID(QAOA):  {qaoa_ro_mu:.4f} ± {qaoa_ro_se:.4f}")

    lines.append("")
    lines.append("Normalized AUC over [0,B] (mean ± stderr):")
    lines.append(f"  AUC Expectation ID(VQE):  {vqe_exp_auc:.4f} ± {vqe_exp_auc_se:.4f}")
    lines.append(f"  AUC Expectation ID(QAOA): {qaoa_exp_auc:.4f} ± {qaoa_exp_auc_se:.4f}")
    lines.append(f"  AUC Readout     ID(VQE):  {vqe_ro_auc:.4f} ± {vqe_ro_auc_se:.4f}")
    lines.append(f"  AUC Readout     ID(QAOA): {qaoa_ro_auc:.4f} ± {qaoa_ro_auc_se:.4f}")

    lines.append("")
    lines.append("Gaps at budget B:")
    lines.append(f"  Δ_expect = VQE - QAOA = {gap_exp:.4f}")
    lines.append(f"  Δ_readout= VQE - QAOA = {gap_ro:.4f}")
    if np.isfinite(gap_closure):
        lines.append(f"  Gap-closure from sampling = 1 - Δ_readout/Δ_expect = {gap_closure:.3f}")

    lines.append("")
    lines.append("Tail factor (readout/expectation) at budget B:")
    lines.append(f"  tail_factor(VQE)  = {tail_factor_vqe:.3f}")
    lines.append(f"  tail_factor(QAOA) = {tail_factor_qaoa:.3f}")

    if tail_mu is not None:
        vqe_tail_mu, qaoa_tail_mu = tail_mu
        vqe_tail_se, qaoa_tail_se = tail_se
        vqe_tail_auc, qaoa_tail_auc = tail_auc_mu
        vqe_tail_auc_se, qaoa_tail_auc_se = tail_auc_se

        lines.append("")
        if str(a.tail_metric).lower() == "hit":
            lines.append(f"Tail metric: hit-rate P(cut >= (1-ε)J*) with ε={a.hit_eps:.3f}")
        else:
            lines.append(f"Tail metric: top-k tail P(cut in top {100*a.topk_frac:.2f}% bitstrings)")
        lines.append(f"  Tail@B ID(VQE):   {vqe_tail_mu:.4f} ± {vqe_tail_se:.4f}")
        lines.append(f"  Tail@B ID(QAOA):  {qaoa_tail_mu:.4f} ± {qaoa_tail_se:.4f}")
        lines.append(f"  AUC Tail ID(VQE): {vqe_tail_auc:.4f} ± {vqe_tail_auc_se:.4f}")
        lines.append(f"  AUC Tail ID(QAOA):{qaoa_tail_auc:.4f} ± {qaoa_tail_auc_se:.4f}")

    if sweep_stats and a.supp_shots.strip() and a.shots_sweep_plot:
        lines.append("")
        lines.append("Readout shot sweep at budget B (best-of-S):")
        for (S, mu_v, se_v, mu_q, se_q) in sweep_stats:
            lines.append(f"  S={S:4d} | VQE {mu_v:.4f} ± {se_v:.4f} | QAOA {mu_q:.4f} ± {se_q:.4f}")

    lines.append("")
    lines.append("Saved outputs:")
    lines.append(f"  - fig2C_expect_and_readout_vs_evals.{a.fmt}")
    if str(a.tail_metric).lower() != "none":
        lines.append(f"  - fig2C_tailprob_vs_evals.{a.fmt}")
    if a.supp_shots.strip() and a.shots_sweep_plot:
        lines.append(f"  - fig2C_readout_shots_sweep.{a.fmt}")
    if a.cost_plot:
        lines.append(f"  - fig2C_evals_vs_outer.{a.fmt}")
    if a.tradeoff_plot:
        lines.append(f"  - fig2C_tradeoff_scatter.{a.fmt}")
    lines.append("  - exp2C_results.csv")
    lines.append("  - exp2C_summary.txt")

    with open(out / "exp2C_summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")

    # --- plots (legend-only; no annotation boxes)
    plot_expect_and_readout_vs_evals(
        out / f"fig2C_expect_and_readout_vs_evals.{a.fmt}",
        curves_expect, curves_readout_main,
        budget=B,
        readout_shots=a.readout_shots,
        annotate=False
    )

    if str(a.tail_metric).lower() != "none" and curves_tail["VQE"]:
        if str(a.tail_metric).lower() == "hit":
            tail_label = rf"Best-so-far $P(\mathrm{{cut}}\geq (1-\epsilon)J^*)$  ($\epsilon$={a.hit_eps:.2f})"
        else:
            tail_label = rf"Best-so-far $P(\mathrm{{cut}}\in \mathrm{{top}}\ {100*a.topk_frac:.1f}\%)$"
        plot_tailprob_vs_evals(
            out / f"fig2C_tailprob_vs_evals.{a.fmt}",
            curves_tail,
            budget=B,
            tail_label=tail_label,
            annotate=False
        )

    if a.supp_shots.strip() and a.shots_sweep_plot:
        shots_sorted = sorted(set(shot_list))
        stats_vqe = [(budget_stats(curves_readout_byS[S], "VQE")) for S in shots_sorted]
        stats_qaoa = [(budget_stats(curves_readout_byS[S], "QAOA")) for S in shots_sorted]
        plot_readout_shots_sweep(
            out / f"fig2C_readout_shots_sweep.{a.fmt}",
            shots_sorted, stats_vqe, stats_qaoa,
            budget=B,
            annotate=False
        )

    if a.cost_plot:
        plot_evals_vs_outer(out / f"fig2C_evals_vs_outer.{a.fmt}", evals_mean)

    if a.tradeoff_plot:
        # scatter points at budget B for each seed/method
        vqe_x = np.array([step_value_at(r["evals"], r["best"], B) for r in curves_expect["VQE"]], float)
        vqe_y = np.array([step_value_at(r["evals"], r["best"], B) for r in curves_readout_main["VQE"]], float)
        qaoa_x = np.array([step_value_at(r["evals"], r["best"], B) for r in curves_expect["QAOA"]], float)
        qaoa_y = np.array([step_value_at(r["evals"], r["best"], B) for r in curves_readout_main["QAOA"]], float)

        plot_tradeoff_scatter(
            out / f"fig2C_tradeoff_scatter.{a.fmt}",
            vqe_x, vqe_y, qaoa_x, qaoa_y,
            xlabel=r"Best expectation at budget $B$  ($J/J^*$)",
            ylabel=rf"Best-of-{int(a.readout_shots)} readout at budget $B$  ($/J^*$)",
            annotate=False
        )

    print("\n".join(lines))
    print("Saved to:", out.resolve())


if __name__ == "__main__":
    main()