#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp0D_inner_budget_ablation.py
==============================

Experiment 0D (Supplement): Inner-budget ablation (iters × restarts)
--------------------------------------------------------------------

Reviewer question:
  "Is the CR-ImpDiff advantage just an artifact of inner-solver convergence?"

What we do:
  Sweep the inner VQE SPSA budget across a grid:
    - inner_iters  (SPSA iterations per restart)
    - restarts     (number of independent SPSA runs; best-of-restarts chosen)
  For each grid cell, run paired outer optimization with:
    - ID        : correlator-reuse implicit differentiation (CR-ImpDiff)
    - FD_VALUE  : black-box finite differences on the VALUE function F(λ)
                 (requires re-solving the inner problem at λ±c per outer step)

Fairness / cost model:
  - We compare at a FIXED total evaluation budget B (energy evaluations).
  - Best-so-far for FD_VALUE is defined over all value queries it actually makes
    (center and perturbed), so we do NOT discard evaluated candidates.

Metrics:
  - AUC_B of best-so-far curve y(b)/J* over b in [0,B], with step-function integral
  - AUC_gain = AUC_ID - AUC_FD_VALUE
  - win_rate = fraction of instances with AUC_gain > 0

Outputs (paper-ready):
  - fig0D_inner_budget_heatmap.<fmt>
      2 panels: (a) mean ΔAUC heatmap, (b) win-rate heatmap
  - runs0D_inner_budget_metrics.csv
      per-instance per-cell metrics
  - table0D_inner_budget_summary.csv / .tex
      per-cell mean ± s.e.m. summary
  - exp0D_summary.txt
      compact text summary (copyable into supplement notes)

Recommended minimal run (not too heavy):
  python exp0D_inner_budget_ablation.py \
    --out out_exp0D \
    --kind periodic --periodic_K 6 \
    --inner_iters_list 14,28,42 \
    --restarts_list 1,2,4 \
    --num_seeds 8 \
    --budget_evals 5100 \
    --shots 0 \
    --fmt pdf

Notes:
  - n=12 keeps the classical diagnostic J* feasible (2^n = 4096).
  - shots=0 means exact expectation evaluation (statevector).
  - shots>0 simulates shot-noisy energy evaluation with that many bitstring samples
    per energy evaluation.

"""

import math
import argparse
import csv
import warnings
import logging
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# ------------------------------------------------------------------------------
# Silence noisy-but-harmless fontTools PDF logs and some numpy warnings
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
# 1) Paper-ish plotting style (same spirit as main script)
# ==============================================================================

COLORS = {
    "ID":  "#D62728",  # red
    "FD":  "#1F77B4",  # blue
    "GT":  "#000000",
}

COL_W, FULL_W = 3.37, 6.95
H_COL = 2.8


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



def plot_inner_budget_heatmap(
    path: Path,
    inner_iters_list: List[int],
    restarts_list: List[int],
    auc_gain_mean: np.ndarray,
    win_rate: np.ndarray,
):
    """
    Paper-ready: 2-panel figure
      (a) mean ΔAUC heatmap, (b) win-rate heatmap
    No titles; includes axis labels and colorbars.
    Fix: panel (b) no longer unreadable (auto text color + better colormap).
    """
    set_pub_style(grid=False)
    fig, axs = plt.subplots(1, 2, figsize=(FULL_W, H_COL), constrained_layout=True)

    # -----------------------------
    # Panel (a): ΔAUC
    # -----------------------------
    ax = axs[0]
    vmax = float(np.max(np.abs(auc_gain_mean))) if np.isfinite(auc_gain_mean).any() else 1.0
    vmax = max(vmax, 1e-6)
    im0 = ax.imshow(
        auc_gain_mean,
        origin="lower",
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=+vmax,
        interpolation="nearest",
    )
    ax.set_xlabel("Inner SPSA iterations")
    ax.set_ylabel("Restarts")
    ax.set_xticks(np.arange(len(inner_iters_list)))
    ax.set_xticklabels([str(x) for x in inner_iters_list])
    ax.set_yticks(np.arange(len(restarts_list)))
    ax.set_yticklabels([str(r) for r in restarts_list])

    # annotate means (also choose readable text color)
    cmap0 = plt.get_cmap("RdBu_r")
    norm0 = mpl.colors.Normalize(vmin=-vmax, vmax=+vmax)
    for iy in range(len(restarts_list)):
        for ix in range(len(inner_iters_list)):
            val = float(auc_gain_mean[iy, ix])
            if np.isfinite(val):
                rgba = cmap0(norm0(val))
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                txt_color = "black" if lum > 0.6 else "white"
                ax.text(ix, iy, f"{val:+.3f}", ha="center", va="center", fontsize=7, color=txt_color)

    cb0 = fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)
    cb0.set_label(r"$\Delta \mathrm{AUC}_B$  (ID $-$ BB-FD)")

    # -----------------------------
    # Panel (b): win rate  (FIXED)
    # -----------------------------
    ax = axs[1]

    # pick a readable colormap (choose ONE)
    cmap_wr = plt.get_cmap("Blues")   # alt: "Greys_r" or "RdBu_r"
    norm_wr = mpl.colors.Normalize(vmin=0.0, vmax=1.0)

    im1 = ax.imshow(
        win_rate,
        origin="lower",
        aspect="auto",
        cmap=cmap_wr,
        norm=norm_wr,
        interpolation="nearest",
    )
    ax.set_xlabel("Inner SPSA iterations")
    ax.set_ylabel("Restarts")
    ax.set_xticks(np.arange(len(inner_iters_list)))
    ax.set_xticklabels([str(x) for x in inner_iters_list])
    ax.set_yticks(np.arange(len(restarts_list)))
    ax.set_yticklabels([str(r) for r in restarts_list])

    for iy in range(len(restarts_list)):
        for ix in range(len(inner_iters_list)):
            val = float(win_rate[iy, ix])
            if np.isfinite(val):
                rgba = cmap_wr(norm_wr(val))
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                txt_color = "black" if lum > 0.6 else "white"
                ax.text(
                    ix, iy, f"{100*val:.0f}%",
                    ha="center", va="center",
                    fontsize=7, color=txt_color
                )

    cb1 = fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
    cb1.set_label("Win rate (ΔAUC>0)")

    # panel labels (small, unobtrusive)
    axs[0].text(0.02, 0.98, "(a)", transform=axs[0].transAxes, ha="left", va="top")
    axs[1].text(0.02, 0.98, "(b)", transform=axs[1].transAxes, ha="left", va="top")

    _savefig(fig, path)
    plt.close(fig)



# ==============================================================================
# 2) Utilities + problem instance + canonical family
# ==============================================================================

def to_uint_seed(seed: int) -> int:
    return int(seed) % (2**32 - 1)


def parse_int_list(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    return [int(x) for x in s.split(",") if x.strip()]


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


class Family1D:
    """
    Canonical families:
      linear:    f = sqrt(3) s x
      quadratic: f = sqrt(45/4) s (x^2 - 1/3)
      periodic:  f = sqrt(2) cos(pi k x + phi)
    All mean-zero and RMS-normalized under x~Unif[-1,1] (periodic in expectation over phi).
    """
    def __init__(self, m: int, kind: str, lam_bounds: Tuple[float, float],
                 rng: np.random.Generator, K: int = 6):
        self.kind = kind
        self.lam_min, self.lam_max = map(float, lam_bounds)
        self.mid = 0.5 * (self.lam_min + self.lam_max)
        self.Delta = max(1e-12, self.lam_max - self.lam_min)
        self.dx_dlam = 2.0 / self.Delta

        # positive baseline + moderate amplitude
        self.wbar = rng.uniform(2.0, 3.0, size=m).astype(float)
        self.A = rng.uniform(0.3, 0.8, size=m).astype(float)

        if kind in ("linear", "quadratic"):
            self.s = rng.choice([-1.0, +1.0], size=m).astype(float)
            self.k = None
            self.phi = None
        elif kind == "periodic":
            self.s = None
            self.k = rng.integers(1, K + 1, size=m).astype(float)
            self.phi = rng.uniform(0.0, 2*np.pi, size=m).astype(float)
        else:
            raise ValueError("kind must be linear/quadratic/periodic")

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
            f = c * self.s * (x*x - 1.0/3.0)
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
        # hard safety
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        return w.astype(float)

    def dw_dlam(self, lam: float) -> np.ndarray:
        _, df = self.f_df(self.x(lam))
        dw = self.A * df * self.dx_dlam
        dw = np.nan_to_num(dw, nan=0.0, posinf=0.0, neginf=0.0)
        return dw.astype(float)


# ==============================================================================
# 3) VQE simulator (statevector) + SPSA inner
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
        # sample computational basis indices
        idx = rng.choice(np.arange(probs.size), size=shots, replace=True, p=probs)
        # cut_mask[idx] -> (shots, m), average -> p_cut
        with np.errstate(all="ignore"):
            p_cut = np.mean(cut_mask[idx, :], axis=0).astype(np.float64)
        np.nan_to_num(p_cut, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        # exact correlators: zexp = ZZ @ probs
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
    gamma: float = 0.101
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
    Run restarts SPSA inner solves and pick best.
    Returns:
      best_params, best_J, best_p_cut, evals_used
    """
    D = init_params.size
    bounds = [(-math.pi, math.pi)] * D

    best_params = init_params.copy()
    best_E = float("inf")
    evals_total = 0

    for r in range(restarts):
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

        p_star, E_star, ev = spsa_minimize(Efun, p0, bounds, iters=inner_iters, seed=seed_base + 17 * (r + 1))
        evals_total += ev

        if np.isfinite(E_star) and E_star < best_E:
            best_E = E_star
            best_params = p_star.copy()

    # evaluate final best to obtain J and p_cut (counts as ONE energy evaluation)
    eval_rng_final = np.random.default_rng(to_uint_seed(seed_base + 424242))
    J, p_cut, _ = vqe_expect(n, best_params, L, w, ZZ_edges, cut_mask, shots, eval_rng_final)
    evals_total += 1

    return best_params, float(J), p_cut.astype(np.float64), int(evals_total)


# ==============================================================================
# 4) Classical diagnostic scale J* and outer loops with budget
# ==============================================================================

def classical_Jstar_max(
    fam: Family1D,
    cut_mask: np.ndarray,
    grid_points: int
) -> float:
    """
    J* = max_{λ in grid} max_z J(z;λ)
    (Used only for normalization / diagnostic.)
    """
    lams = np.linspace(fam.lam_min, fam.lam_max, int(grid_points))
    J_star = -1e30
    for lam in lams:
        w = fam.w(float(lam)).astype(np.float64)
        with np.errstate(all="ignore"):
            vals = cut_mask @ w
        vals = np.nan_to_num(vals, nan=-1e30, posinf=-1e30, neginf=-1e30)
        J_star = max(J_star, float(np.max(vals)))
    if not np.isfinite(J_star):
        J_star = 1.0
    return float(J_star)


def run_outer_budget(
    mode: str,
    n: int,
    edges,
    fam: Family1D,
    ZZ_edges: np.ndarray,
    cut_mask: np.ndarray,
    lam0: float,
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
    Jstar: float
) -> Dict[str, np.ndarray]:
    """
    Runs outer optimization until budget or outer_max.
    Returns a trace of (evals, best_norm) at each value-query completion.
    """
    lam_min, lam_max = fam.lam_min, fam.lam_max
    lam = float(np.clip(lam0, lam_min, lam_max))

    D = 2 * n * L
    params = np.zeros(D, float)

    evals = 0.0
    best = -1e30

    events_evals = []
    events_best = []

    c = float(c_frac * (lam_max - lam_min))

    for t in range(1, int(outer_max) + 1):
        # center value query F(lam) via inner solve
        w = fam.w(lam)

        params, J, p_cut, ev_in = inner_solve(
            n=n, L=L, w=w, ZZ_edges=ZZ_edges, cut_mask=cut_mask,
            init_params=params, inner_iters=inner_iters, restarts=restarts,
            seed_base=seed + 100000 * t + 111, shots=shots
        )
        evals += float(ev_in)
        best = max(best, float(J))
        events_evals.append(evals)
        events_best.append(best / Jstar)

        # outer signal
        if mode == "ID":
            g = float(np.dot(fam.dw_dlam(lam), p_cut))
        elif mode == "FD_VALUE":
            lp = float(np.clip(lam + c, lam_min, lam_max))
            lm = float(np.clip(lam - c, lam_min, lam_max))

            # +c value query
            w_p = fam.w(lp)
            p_p, Jp, _, evp = inner_solve(
                n=n, L=L, w=w_p, ZZ_edges=ZZ_edges, cut_mask=cut_mask,
                init_params=params, inner_iters=inner_iters, restarts=restarts,
                seed_base=seed + 100000 * t + 777, shots=shots
            )
            evals += float(evp)
            best = max(best, float(Jp))
            events_evals.append(evals)
            events_best.append(best / Jstar)

            # -c value query
            w_m = fam.w(lm)
            p_m, Jm, _, evm = inner_solve(
                n=n, L=L, w=w_m, ZZ_edges=ZZ_edges, cut_mask=cut_mask,
                init_params=params, inner_iters=inner_iters, restarts=restarts,
                seed_base=seed + 100000 * t + 999, shots=shots
            )
            evals += float(evm)
            best = max(best, float(Jm))
            events_evals.append(evals)
            events_best.append(best / Jstar)

            # gradient
            g = (float(Jp) - float(Jm)) / (2.0 * c) if c > 0 else 0.0

            # optionally: warm-start next center at the better perturbed params
            # (keeps baseline strong; but also increases coupling.)
            # We'll keep params as the last center solution for simplicity.
            _ = p_p, p_m
        else:
            raise ValueError("mode must be ID or FD_VALUE")

        # outer update
        eta = float(eta0 / (t ** eta_pow))
        step = float(eta * g)
        step = float(np.clip(step, -step_clip, step_clip))
        lam = float(np.clip(lam + step, lam_min, lam_max))

        # budget stop (we allow slight overshoot; AUC truncation will fix)
        if evals >= float(budget_evals):
            break

    return {
        "evals": np.asarray(events_evals, dtype=float),
        "best_norm": np.asarray(events_best, dtype=float),
        "evals_end": np.array([evals], dtype=float),
    }


def auc_step(evals: np.ndarray, values: np.ndarray, budget: float) -> float:
    """
    Step-function AUC on [0, budget].
    Assumes best-so-far is held constant between recorded events.
    """
    evals = np.asarray(evals, float)
    values = np.asarray(values, float)
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
        # nothing before budget -> constant at first value
        return float(values[0])

    if evals[-1] < budget:
        evals = np.append(evals, budget)
        values = np.append(values, values[-1])

    # step integral: sum (Δx * y_i)
    dx = np.diff(evals)
    area = float(np.sum(dx * values[:-1]))
    return area / float(budget)


# ==============================================================================
# 5) I/O helpers: CSV + LaTeX table
# ==============================================================================

def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_table_tex(path: Path, rows: List[Dict], caption: str, label: str):
    """
    Minimal booktabs LaTeX table:
      restarts, inner_iters, mean_auc_gain ± sem, win_rate
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{ccc c}\n")
        f.write("\\toprule\n")
        f.write("Restarts & Inner iters & $\\Delta\\mathrm{AUC}_B$ (mean $\\pm$ s.e.m.) & Win rate \\\\\n")
        f.write("\\midrule\n")
        for r in rows:
            f.write(f"{r['restarts']} & {r['inner_iters']} & "
                    f"{r['auc_gain_mean']:+.4f} $\\pm$ {r['auc_gain_sem']:.4f} & "
                    f"{100.0*r['win_rate']:.0f}\\% \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


# ==============================================================================
# 6) Main
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="experiment5")
    p.add_argument("--fmt", type=str, default="pdf", choices=["pdf", "png", "svg"])

    # instance / family
    p.add_argument("--seed0", type=int, default=7)
    p.add_argument("--num_seeds", type=int, default=7)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--p_edge", type=float, default=0.45)
    p.add_argument("--kind", type=str, default="periodic", choices=["linear", "quadratic", "periodic"])
    p.add_argument("--periodic_K", type=int, default=6)
    p.add_argument("--lam_min", type=float, default=-5.0)
    p.add_argument("--lam_max", type=float, default=5.0)
    p.add_argument("--lam0", type=float, default=0.8)

    # VQE / inner
    p.add_argument("--L", type=int, default=2)
    p.add_argument("--shots", type=int, default=0, help="shots per energy evaluation (0 = exact)")

    # ablation grid
    p.add_argument("--inner_iters_list", type=str, default="14,28,42")
    p.add_argument("--restarts_list", type=str, default="1,2,4")

    # outer
    p.add_argument("--outer_max", type=int, default=200)
    p.add_argument("--eta0", type=float, default=0.35)
    p.add_argument("--eta_pow", type=float, default=0.6)
    p.add_argument("--step_clip", type=float, default=0.6)
    p.add_argument("--c_frac", type=float, default=0.05)

    # budget + J* grid
    p.add_argument("--budget_evals", type=float, default=5100.0)
    p.add_argument("--Jstar_grid", type=int, default=401)

    return p.parse_args()


def main():
    a = parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    inner_iters_list = parse_int_list(a.inner_iters_list)
    restarts_list = parse_int_list(a.restarts_list)
    if not inner_iters_list or not restarts_list:
        raise ValueError("inner_iters_list and restarts_list must be non-empty.")

    # collect per-run rows
    run_rows = []

    # for heatmaps: store per-cell per-seed values, then aggregate
    # shape: (R, I, S)
    Rn = len(restarts_list)
    In = len(inner_iters_list)
    Sn = int(a.num_seeds)

    auc_gain = np.full((Rn, In, Sn), np.nan, dtype=float)
    win = np.zeros((Rn, In, Sn), dtype=float)

    for s_idx in range(Sn):
        seed = int(a.seed0 + s_idx)
        rng = np.random.default_rng(to_uint_seed(seed))

        # instance
        edges = generate_random_graph(a.n, a.p_edge, rng)
        if not edges:
            # resample with a deterministic bump to avoid empty graphs
            rng2 = np.random.default_rng(to_uint_seed(seed + 12345))
            edges = generate_random_graph(a.n, a.p_edge, rng2)
        if not edges:
            raise RuntimeError("Graph has 0 edges; increase p_edge or change seed0.")

        Z = precompute_z_big_endian(a.n)
        cut_mask = build_cut_mask(edges, Z)
        ZZ_edges = build_ZZ_edges(edges, Z)

        fam = Family1D(
            m=len(edges),
            kind=a.kind,
            lam_bounds=(a.lam_min, a.lam_max),
            rng=rng,
            K=a.periodic_K
        )

        # diagnostic normalization J*
        Jstar = classical_Jstar_max(fam, cut_mask, a.Jstar_grid)
        if Jstar <= 0 or not np.isfinite(Jstar):
            Jstar = 1.0

        for iy, restarts in enumerate(restarts_list):
            for ix, inner_iters in enumerate(inner_iters_list):
                # ID run
                hist_id = run_outer_budget(
                    mode="ID",
                    n=a.n, edges=edges, fam=fam, ZZ_edges=ZZ_edges, cut_mask=cut_mask,
                    lam0=a.lam0,
                    outer_max=a.outer_max,
                    inner_iters=inner_iters,
                    restarts=restarts,
                    L=a.L,
                    seed=seed,
                    eta0=a.eta0,
                    eta_pow=a.eta_pow,
                    step_clip=a.step_clip,
                    c_frac=a.c_frac,
                    shots=a.shots,
                    budget_evals=a.budget_evals,
                    Jstar=Jstar
                )

                # FD run
                hist_fd = run_outer_budget(
                    mode="FD_VALUE",
                    n=a.n, edges=edges, fam=fam, ZZ_edges=ZZ_edges, cut_mask=cut_mask,
                    lam0=a.lam0,
                    outer_max=a.outer_max,
                    inner_iters=inner_iters,
                    restarts=restarts,
                    L=a.L,
                    seed=seed,
                    eta0=a.eta0,
                    eta_pow=a.eta_pow,
                    step_clip=a.step_clip,
                    c_frac=a.c_frac,
                    shots=a.shots,
                    budget_evals=a.budget_evals,
                    Jstar=Jstar
                )

                auc_id = auc_step(hist_id["evals"], hist_id["best_norm"], a.budget_evals)
                auc_fd = auc_step(hist_fd["evals"], hist_fd["best_norm"], a.budget_evals)
                gain = float(auc_id - auc_fd)

                auc_gain[iy, ix, s_idx] = gain
                win[iy, ix, s_idx] = 1.0 if gain > 0 else 0.0

                run_rows.append({
                    "seed": seed,
                    "kind": a.kind,
                    "periodic_K": a.periodic_K,
                    "n": a.n,
                    "p_edge": a.p_edge,
                    "shots": a.shots,
                    "budget_evals": a.budget_evals,
                    "inner_iters": inner_iters,
                    "restarts": restarts,
                    "Jstar": Jstar,
                    "auc_id": auc_id,
                    "auc_fd": auc_fd,
                    "auc_gain": gain,
                    "evals_end_id": float(hist_id["evals_end"][0]),
                    "evals_end_fd": float(hist_fd["evals_end"][0]),
                })

                print(f"[seed={seed:3d}] iters={inner_iters:3d} restarts={restarts:2d} "
                      f"AUC_ID={auc_id:.4f} AUC_FD={auc_fd:.4f} gain={gain:+.4f}")

    # save per-run CSV
    runs_csv = out / "runs0D_inner_budget_metrics.csv"
    write_csv(
        runs_csv,
        run_rows,
        fieldnames=list(run_rows[0].keys()) if run_rows else []
    )

    # aggregate per-cell
    summary_rows = []
    auc_gain_mean = np.full((Rn, In), np.nan, dtype=float)
    win_rate = np.full((Rn, In), np.nan, dtype=float)

    for iy, restarts in enumerate(restarts_list):
        for ix, inner_iters in enumerate(inner_iters_list):
            vals = auc_gain[iy, ix, :]
            vals = vals[np.isfinite(vals)]
            N = int(vals.size)
            if N == 0:
                mean = float("nan")
                sem = float("nan")
                wr = float("nan")
            else:
                mean = float(np.mean(vals))
                sem = float(np.std(vals, ddof=1) / math.sqrt(N)) if N > 1 else float("nan")
                wr = float(np.mean(win[iy, ix, :]))

            auc_gain_mean[iy, ix] = mean
            win_rate[iy, ix] = wr

            summary_rows.append({
                "restarts": int(restarts),
                "inner_iters": int(inner_iters),
                "N": int(N),
                "auc_gain_mean": mean,
                "auc_gain_sem": sem,
                "win_rate": wr,
            })

    # save summary table
    summary_csv = out / "table0D_inner_budget_summary.csv"
    write_csv(
        summary_csv,
        summary_rows,
        fieldnames=["restarts", "inner_iters", "N", "auc_gain_mean", "auc_gain_sem", "win_rate"]
    )
    summary_tex = out / "table0D_inner_budget_summary.tex"
    write_table_tex(
        summary_tex,
        summary_rows,
        caption=("0D inner-budget ablation (periodic family by default). "
                 "Each cell reports mean±s.e.m. of ΔAUC_B = AUC_ID − AUC_BB-FD "
                 "at fixed evaluation budget B, over paired random instances."),
        label="tab:0D_inner_budget"
    )

    # figure
    fig_path = out / f"fig0D_inner_budget_heatmap.{a.fmt}"
    plot_inner_budget_heatmap(
        fig_path,
        inner_iters_list=inner_iters_list,
        restarts_list=restarts_list,
        auc_gain_mean=auc_gain_mean,
        win_rate=win_rate
    )

    # compact text summary
    txt = out / "exp0D_summary.txt"
    with txt.open("w", encoding="utf-8") as f:
        f.write("Experiment 0D — Inner-budget ablation\n")
        f.write(f"kind={a.kind} | periodic_K={a.periodic_K} | n={a.n} | p_edge={a.p_edge}\n")
        f.write(f"shots={a.shots} | budget_evals={a.budget_evals} | seeds={a.num_seeds}\n")
        f.write(f"inner_iters_list={inner_iters_list}\n")
        f.write(f"restarts_list={restarts_list}\n\n")
        # global summary
        all_g = auc_gain.reshape(-1, Sn)
        flat = all_g[np.isfinite(all_g)]
        if flat.size:
            f.write(f"Overall mean ΔAUC over all cells: {float(np.mean(flat)):+.4f}\n")
            f.write(f"Overall win rate over all cells: {float(np.mean(flat > 0)) * 100:.1f}%\n")
        f.write("\nPer-cell summary (restarts, iters, mean±sem, win_rate):\n")
        for r in summary_rows:
            f.write(f"  r={r['restarts']}, it={r['inner_iters']}: "
                    f"{r['auc_gain_mean']:+.4f} ± {r['auc_gain_sem']:.4f}, "
                    f"{100.0*r['win_rate']:.0f}%\n")

    print("\nSaved to:", out.resolve())
    print("Figure:", fig_path.name)
    print("Runs:", runs_csv.name)
    print("Table:", summary_tex.name)


if __name__ == "__main__":
    main()
