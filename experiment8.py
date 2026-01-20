#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp2D_heatmap_id_vs_fd_np_graphclasses.py
================================================

Experiment 2D (new): (n, p) heatmaps showing where Implicit Differentiation (ID)
beats Black-Box Finite Differences (BB-FD) under a matched *energy-evaluation*
budget, across multiple graph classes.

What this is meant to show (ICML-friendly story)
------------------------------------------------
- In bilevel Max-Cut we optimize F(λ) = max_ϑ J(ϑ, λ).
- A naive outer hypergradient estimate via value probing / finite differences
  (BB-FD) needs *two additional* evaluations of F(λ±c). In a true bilevel
  setting, that implies *two additional inner solves* per outer step.
- ID (correlator reuse / envelope) avoids those extra probes and therefore
  makes better progress for the same evaluation budget.

This script quantifies that effect on a grid of Erdős–Rényi parameters G(n,p)
(and optionally other graph generators mapped to the same expected degree), and
summarizes the advantage in a heatmap.

Primary metric
--------------
ΔAUC_B = AUC_B(ID) - AUC_B(BB-FD)
where AUC_B is the normalized area under the best-so-far curve of J/J* vs.
energy-evaluation budget B.

Each heatmap cell also annotates the win-rate (fraction of seeds with ΔAUC_B>0).

Outputs (in --out)
------------------
- exp2D_rows.csv                     (per-run raw results)
- exp2D_agg.csv                      (per-cell aggregates)
- fig_exp2D_heatmap_deltaAUC.<fmt>   (multi-panel heatmap)
- exp2D_summary.txt

Example
-------
python exp2D_heatmap_id_vs_fd_np_graphclasses.py \
  --graph_classes er,ring,ws,ba \
  --n_list 8,10,12 \
  --p_list 0.20,0.35,0.50 \
  --seeds 1,2,3,4,5 \
  --budget_evals 6000 \
  --kind periodic --periodic_K 6 \
  --inner 60 --outer_max 200 \
  --fmt pdf

Notes
-----
- This is a pure statevector experiment, so n should be kept modest.
- "p" in --p_list corresponds to edge probability for ER graphs. For other
  graph classes we map p to a target expected degree d = p*(n-1).
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
# 1) Minimal publication plotting (Nature-ish)
# ==============================================================================

COL_W = 6.95  # inches (two-column)


def set_pub_style(grid: bool = False):
    mpl.rcdefaults()
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Liberation Serif"],
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "lines.linewidth": 1.2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": False,
        "ytick.right": False,
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


def parse_float_list(s: str) -> List[float]:
    s = (s or "").strip()
    if not s:
        return []
    out = []
    for chunk in s.split(","):
        chunk = chunk.strip()
        if chunk:
            out.append(float(chunk))
    return out


def step_auc(evals: np.ndarray, y_step: np.ndarray, x_max: float) -> float:
    """AUC of a step function y(evals) over [0, x_max] (right-continuous)."""
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
# 3) Graph generators (graph classes)
# ==============================================================================


def _edge_key(i: int, j: int) -> Tuple[int, int]:
    return (i, j) if i < j else (j, i)


def generate_er_graph(n: int, p_edge: float, rng: np.random.Generator) -> List[Tuple[int, int]]:
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p_edge:
                edges.append((i, j))
    return edges


def generate_regular_ring(n: int, k: int) -> List[Tuple[int, int]]:
    """Undirected ring-lattice with degree k (k must be even)."""
    k = int(k)
    k = max(0, min(k, n - 1))
    if k % 2 == 1:
        k -= 1
    if k <= 0:
        return []
    half = k // 2
    edges = set()
    for i in range(n):
        for d in range(1, half + 1):
            j = (i + d) % n
            edges.add(_edge_key(i, j))
    return sorted(edges)


def generate_watts_strogatz(n: int, k: int, beta: float, rng: np.random.Generator) -> List[Tuple[int, int]]:
    """Watts–Strogatz small-world: start from ring lattice, rewire each edge with prob beta."""
    beta = float(np.clip(beta, 0.0, 1.0))
    edges = generate_regular_ring(n, k)
    if not edges or beta <= 0:
        return edges

    # adjacency for fast checks
    adj = {i: set() for i in range(n)}
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)

    new_edges = set(edges)

    for (i, j) in list(edges):
        if rng.random() >= beta:
            continue
        # rewire endpoint j -> k_new (keep i fixed)
        # remove old edge
        new_edges.discard(_edge_key(i, j))
        adj[i].discard(j)
        adj[j].discard(i)

        # choose new node not equal i and not already connected
        candidates = [x for x in range(n) if x != i and x not in adj[i]]
        if not candidates:
            # revert
            new_edges.add(_edge_key(i, j))
            adj[i].add(j)
            adj[j].add(i)
            continue
        k_new = int(rng.choice(candidates))
        new_edges.add(_edge_key(i, k_new))
        adj[i].add(k_new)
        adj[k_new].add(i)

    return sorted(new_edges)


def generate_barabasi_albert(n: int, m: int, rng: np.random.Generator) -> List[Tuple[int, int]]:
    """Simple Barabási–Albert preferential attachment graph."""
    n = int(n)
    m = int(m)
    if n <= 1:
        return []
    m = max(1, min(m, n - 1))

    # start with a complete graph on m+1 nodes
    m0 = min(n, m + 1)
    edges = set()
    degree = np.zeros(n, dtype=int)

    for i in range(m0):
        for j in range(i + 1, m0):
            edges.add((i, j))
            degree[i] += 1
            degree[j] += 1

    # list of nodes repeated by degree for sampling
    # (works fine for small n)
    repeated = []
    for i in range(m0):
        repeated.extend([i] * degree[i])

    for new in range(m0, n):
        targets = set()
        # if all degrees are zero (shouldn't happen), fall back to uniform
        while len(targets) < min(m, new):
            if repeated:
                t = int(rng.choice(repeated))
            else:
                t = int(rng.integers(0, new))
            targets.add(t)

        for t in targets:
            edges.add(_edge_key(new, t))
            degree[new] += 1
            degree[t] += 1
            repeated.append(new)
            repeated.append(t)

    return sorted(edges)


def generate_graph(graph_class: str, n: int, p: float, rng: np.random.Generator,
                   ws_beta: float = 0.3) -> List[Tuple[int, int]]:
    """Generate edges for a given graph_class.

    p is interpreted as:
      - ER: edge probability
      - ring/ws/ba: mapped to expected degree d = p*(n-1)
    """
    graph_class = str(graph_class).strip().lower()
    p = float(np.clip(p, 0.0, 1.0))

    if graph_class in ("er", "erdos", "erdos_renyi", "gnp"):
        return generate_er_graph(n, p, rng)

    # map to target expected degree
    d = float(p * (n - 1))

    if graph_class in ("ring", "regular", "regular_ring"):
        k = int(round(d))
        if k % 2 == 1:
            k += 1
        return generate_regular_ring(n, k)

    if graph_class in ("ws", "watts", "watts_strogatz"):
        k = int(round(d))
        if k % 2 == 1:
            k += 1
        return generate_watts_strogatz(n, k, ws_beta, rng)

    if graph_class in ("ba", "barabasi", "barabasi_albert"):
        m = int(round(d / 2.0))
        m = max(1, min(m, n - 1))
        return generate_barabasi_albert(n, m, rng)

    raise ValueError(f"Unknown graph_class: {graph_class}")


# ==============================================================================
# 4) Precompute cut masks
# ==============================================================================


def precompute_z_big_endian(n: int) -> np.ndarray:
    """Z[q, x] = ±1 eigenvalue of Z on qubit q for computational basis state index x."""
    K = 1 << n
    idx = np.arange(K, dtype=np.uint32)
    Z = np.empty((n, K), dtype=np.int8)
    for q in range(n):
        bitpos = n - 1 - q
        Z[q] = 1 - 2 * ((idx >> bitpos) & 1).astype(np.int8)
    return Z


def build_cut_mask(edges: List[Tuple[int, int]], Z: np.ndarray) -> np.ndarray:
    """cut_mask[x, e] = 1 if edge e is cut by bitstring x, else 0."""
    K = Z.shape[1]
    m = len(edges)
    cut = np.empty((K, m), dtype=np.float64)
    for e, (i, j) in enumerate(edges):
        cut[:, e] = 0.5 * (1.0 - (Z[i] * Z[j]).astype(np.float64))
    return cut


# ==============================================================================
# 5) Parametric weight family w(λ)
# ==============================================================================


class Family1D:
    """w_e(λ) = w̄_e + A_e f_e(x), x = 2(λ-mid)/Δ ∈ [-1,1]."""

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
# 6) Statevector simulator primitives
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
# 7) Expectation value via cut-mask
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
    """Return (J, p_cut, probs)."""
    probs = probs_from_state(psi)
    p_cut = probs @ cut_mask
    J = float(p_cut @ w)
    if not np.isfinite(J):
        J = 0.0
    return float(J), p_cut.astype(np.float64), probs


# ==============================================================================
# 8) VQE ansatz (hardware-efficient)
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
# 9) Inner optimizer: SPSA
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
# 10) Classical envelope maximum for normalization J*
# ==============================================================================


def classical_Jstar(fam: Family1D, cut_mask: np.ndarray, grid_points: int) -> Tuple[float, float]:
    """Approximate J* = max_{λ,z} J(z;λ) by scanning λ on a grid."""
    lams = np.linspace(fam.lam_min, fam.lam_max, int(grid_points))
    bestJ = -1e30
    bestLam = float(lams[0])
    for lam in lams:
        w = fam.w(float(lam)).astype(np.float64)
        cut_vals = cut_mask @ w
        j = float(np.max(cut_vals))
        if j > bestJ:
            bestJ = j
            bestLam = float(lam)
    if not np.isfinite(bestJ) or bestJ <= 0:
        bestJ = 1.0
    return float(bestJ), float(bestLam)


# ==============================================================================
# 11) Outer loops (ID vs BB-FD), budgeted
# ==============================================================================


def run_outer_vqe_id_budgeted(
    n: int,
    cut_mask: np.ndarray,
    fam: Family1D,
    *,
    lam0: float,
    outer_max: int,
    inner: int,
    budget_evals: float,
    eta0: float,
    eta_pow: float,
    step_clip: float,
    seed: int,
    L_vqe: int,
) -> Dict[str, np.ndarray]:
    """ID outer optimization (reuse hypergradient) until budget is exhausted."""

    lam_min, lam_max = fam.lam_min, fam.lam_max
    lam = float(np.clip(lam0, lam_min, lam_max))

    D = 2 * n * L_vqe
    params = np.zeros(D, float)
    bounds = [(-math.pi, math.pi)] * D

    evals = 0.0
    bestJ = -1e18

    hist_evals = []
    hist_best = []

    for t in range(1, int(outer_max) + 1):
        if evals >= budget_evals:
            break

        w = fam.w(lam)

        def Efun(pvec):
            return vqe_energy(n, cut_mask, w, pvec, L_vqe)

        params, _bestE, ev_in = spsa_minimize(Efun, params, bounds, iters=inner, seed=seed + 1000 * t)
        evals += float(ev_in)

        psi = vqe_state(n, params, L_vqe)
        J, p_cut, _probs = expect_J(psi, cut_mask, w)
        evals += 1.0
        bestJ = max(bestJ, float(J))

        hist_evals.append(float(evals))
        hist_best.append(float(bestJ))

        # reuse hypergradient
        g = float(fam.dw_dlam(lam) @ p_cut)

        eta = eta0 / (t ** eta_pow)
        step = float(np.clip(eta * g, -step_clip, step_clip))
        lam = float(np.clip(lam + step, lam_min, lam_max))

    return {
        "evals_cum": np.asarray(hist_evals, float),
        "J_best": np.asarray(hist_best, float),
    }



def run_outer_vqe_bbfd_budgeted(
    n: int,
    cut_mask: np.ndarray,
    fam: Family1D,
    *,
    lam0: float,
    outer_max: int,
    inner: int,
    budget_evals: float,
    eta0: float,
    eta_pow: float,
    step_clip: float,
    seed: int,
    L_vqe: int,
    fd_c_frac: float,
) -> Dict[str, np.ndarray]:
    """Black-box bilevel FD via value-probing F(λ±c): 2 extra inner solves per step."""

    lam_min, lam_max = fam.lam_min, fam.lam_max
    lam = float(np.clip(lam0, lam_min, lam_max))

    D = 2 * n * L_vqe
    params = np.zeros(D, float)
    bounds = [(-math.pi, math.pi)] * D

    c_fd = float(fd_c_frac * (lam_max - lam_min))

    evals = 0.0
    bestJ = -1e18

    hist_evals = []
    hist_best = []

    for t in range(1, int(outer_max) + 1):
        if evals >= budget_evals:
            break

        # ---- inner solve at current λ ----
        w0 = fam.w(lam)

        def Efun0(pvec):
            return vqe_energy(n, cut_mask, w0, pvec, L_vqe)

        params, _bestE, ev0 = spsa_minimize(Efun0, params, bounds, iters=inner, seed=seed + 1000 * t + 0)
        evals += float(ev0)

        psi0 = vqe_state(n, params, L_vqe)
        J0, _p_cut0, _probs0 = expect_J(psi0, cut_mask, w0)
        evals += 1.0
        bestJ = max(bestJ, float(J0))
        hist_evals.append(float(evals))
        hist_best.append(float(bestJ))

        # ---- probes at λ±c (each with its own inner solve) ----
        lp = float(np.clip(lam + c_fd, lam_min, lam_max))
        lm = float(np.clip(lam - c_fd, lam_min, lam_max))

        if abs(lp - lm) < 1e-12:
            g = 0.0
        else:
            # warm-start from current params
            params_p = params.copy()
            params_m = params.copy()

            # plus
            wp = fam.w(lp)

            def Efunp(pvec):
                return vqe_energy(n, cut_mask, wp, pvec, L_vqe)

            params_p, _bestE_p, evp = spsa_minimize(Efunp, params_p, bounds, iters=inner, seed=seed + 1000 * t + 111)
            evals += float(evp)
            psip = vqe_state(n, params_p, L_vqe)
            Jp, _p_cut_p, _probs_p = expect_J(psip, cut_mask, wp)
            evals += 1.0
            bestJ = max(bestJ, float(Jp))
            hist_evals.append(float(evals))
            hist_best.append(float(bestJ))

            # minus
            wm = fam.w(lm)

            def Efunm(pvec):
                return vqe_energy(n, cut_mask, wm, pvec, L_vqe)

            params_m, _bestE_m, evm = spsa_minimize(Efunm, params_m, bounds, iters=inner, seed=seed + 1000 * t + 222)
            evals += float(evm)
            psim = vqe_state(n, params_m, L_vqe)
            Jm, _p_cut_m, _probs_m = expect_J(psim, cut_mask, wm)
            evals += 1.0
            bestJ = max(bestJ, float(Jm))
            hist_evals.append(float(evals))
            hist_best.append(float(bestJ))

            g = float((Jp - Jm) / (lp - lm))

        # ---- outer step ----
        eta = eta0 / (t ** eta_pow)
        step = float(np.clip(eta * g, -step_clip, step_clip))
        lam = float(np.clip(lam + step, lam_min, lam_max))

    return {
        "evals_cum": np.asarray(hist_evals, float),
        "J_best": np.asarray(hist_best, float),
    }


# ==============================================================================
# 12) Heatmap plotting
# ==============================================================================


def _pretty_graph_name(gc: str) -> str:
    gc = str(gc).lower().strip()
    if gc in ("er", "erdos", "erdos_renyi", "gnp"):
        return "Erdos-Renyi"
    if gc in ("ring", "regular", "regular_ring"):
        return "Regular ring"
    if gc in ("ws", "watts", "watts_strogatz"):
        return "Watts-Strogatz"
    if gc in ("ba", "barabasi", "barabasi_albert"):
        return "Barabasi-Albert"
    return gc


def plot_heatmaps(path: Path,
                  n_list: List[int],
                  p_list: List[float],
                  graph_classes: List[str],
                  delta_mats: Dict[str, np.ndarray],
                  win_mats: Dict[str, np.ndarray],
                  *,
                  fmt: str,
                  annotate: bool = True,
                  cmap: str = "RdBu_r"):
    """Multi-panel heatmap grid with shared colorbar.
    Layout rule:
      - 1 panel  -> 1x1
      - 2-4      -> 2 columns (=> 2x2 for 4 panels)
      - >4       -> 3 columns (fallback)
    """

    set_pub_style(grid=False)

    n_panels = len(graph_classes)
    if n_panels <= 0:
        raise ValueError("plot_heatmaps: graph_classes must not be empty")

    # ---- Layout: force 2x2 for the common case of 4 graph classes ----
    if n_panels == 1:
        rows, cols = 1, 1
    elif n_panels <= 4:
        cols = 2
        rows = int(math.ceil(n_panels / cols))
    else:
        cols = 3
        rows = int(math.ceil(n_panels / cols))

    # size heuristic
    fig_h = 2.35 * rows
    fig_w = COL_W if cols > 1 else COL_W  # keep your paper width constant

    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), constrained_layout=True)

    # normalize axes to a 2D array of shape (rows, cols)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    axes = np.asarray(axes).reshape(rows, cols)

    # symmetric color scale across all panels
    all_vals = []
    for gc in graph_classes:
        M = np.asarray(delta_mats[gc], float)
        all_vals.append(M[np.isfinite(M)])
    all_vals = np.concatenate(all_vals) if all_vals else np.asarray([0.0])
    vmax = float(np.max(np.abs(all_vals))) if all_vals.size else 1.0
    vmax = max(vmax, 1e-6)
    vmin = -vmax

    last_im = None
    panel_axes = []

    for idx, gc in enumerate(graph_classes):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]
        panel_axes.append(ax)

        D = np.asarray(delta_mats[gc], float)
        W = np.asarray(win_mats[gc], float)

        last_im = ax.imshow(
            D,
            origin="lower",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            interpolation="nearest",
        )

        # ticks
        ax.set_xticks(np.arange(len(p_list)))
        ax.set_yticks(np.arange(len(n_list)))
        ax.set_xticklabels([f"{p:.2f}" for p in p_list])
        ax.set_yticklabels([str(n) for n in n_list])

        # only bottom row gets x-label
        if r == rows - 1:
            ax.set_xlabel(r"$p$")
        else:
            ax.set_xlabel("")

        # only left col gets y-label
        if c == 0:
            ax.set_ylabel(r"$n$")
        else:
            ax.set_ylabel("")

        ax.set_title(_pretty_graph_name(gc))

        # subtle cell grid
        ax.set_xticks(np.arange(-0.5, len(p_list), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(n_list), 1), minor=True)
        ax.grid(which="minor", linestyle="-", linewidth=0.3, alpha=0.25)
        ax.tick_params(which="minor", bottom=False, left=False)

        # annotations
        if annotate:
            for i in range(len(n_list)):
                for j in range(len(p_list)):
                    d = D[i, j]
                    w = W[i, j]
                    if not np.isfinite(d):
                        txt = "—"
                    else:
                        txt = f"{d:+.2f}\n{100*w:.0f}%"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=7)

    # hide empty axes
    for k in range(n_panels, rows * cols):
        r = k // cols
        c = k % cols
        axes[r, c].axis("off")

    # shared colorbar (attach only to used panels, not the hidden axes)
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=panel_axes, shrink=0.92, pad=0.02)
        cbar.set_label(r"$\Delta\,\mathrm{AUC}_B$ (ID $-$ BB-FD)")

    _savefig(fig, path)


# ==============================================================================
# 13) Main
# ==============================================================================


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--out", type=str, default="out_exp2D_heatmap")
    p.add_argument("--fmt", type=str, default="pdf", choices=["pdf", "png", "svg"])

    # grid over G(n,p)
    p.add_argument("--n_list", type=str, default="8,10,12")
    p.add_argument("--p_list", type=str, default="0.20,0.35,0.50")

    # graph classes
    p.add_argument("--graph_classes", type=str, default="er,ring,ws,ba",
                   help="Comma-separated: er, ring, ws, ba")
    p.add_argument("--ws_beta", type=float, default=0.30,
                   help="Rewiring probability for Watts-Strogatz (density is still set by p).")

    # seeds
    p.add_argument("--seeds", type=str, default="1,2,3")

    # family
    p.add_argument("--kind", type=str, default="periodic", choices=["linear", "quadratic", "periodic"])
    p.add_argument("--periodic_K", type=int, default=6)
    p.add_argument("--lam_min", type=float, default=-5.0)
    p.add_argument("--lam_max", type=float, default=5.0)
    p.add_argument("--lam0", type=float, default=4.0)
    p.add_argument("--lam_grid", type=int, default=301)

    # VQE depth
    p.add_argument("--L_vqe", type=int, default=2)

    # optimization budget and schedules
    p.add_argument("--budget_evals", type=float, default=6000.0)
    p.add_argument("--outer_max", type=int, default=250)
    p.add_argument("--inner", type=int, default=28)

    p.add_argument("--eta0", type=float, default=0.25)
    p.add_argument("--eta_pow", type=float, default=0.4)
    p.add_argument("--step_clip", type=float, default=0.6)

    # FD step
    p.add_argument("--fd_c_frac", type=float, default=0.10)

    # heatmap look
    p.add_argument("--no_annotate", action="store_true", help="Disable per-cell text annotations.")
    p.add_argument("--cmap", type=str, default="RdBu_r", help="Matplotlib colormap name.")

    return p.parse_args()



def main():
    a = parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    n_list = parse_int_list(a.n_list)
    p_list = parse_float_list(a.p_list)
    seeds = parse_int_list(a.seeds)

    if not n_list:
        raise ValueError("--n_list must not be empty")
    if not p_list:
        raise ValueError("--p_list must not be empty")
    if not seeds:
        raise ValueError("--seeds must not be empty")

    graph_classes = [g.strip() for g in (a.graph_classes or "").split(",") if g.strip()]
    if not graph_classes:
        raise ValueError("--graph_classes must not be empty")

    # cache Z by n
    Z_cache: Dict[int, np.ndarray] = {}

    # per-class matrices
    delta_mats: Dict[str, np.ndarray] = {}
    win_mats: Dict[str, np.ndarray] = {}

    # raw rows
    rows = []

    # aggregates
    agg_rows = []

    B = float(a.budget_evals)

    for gc in graph_classes:
        D = np.full((len(n_list), len(p_list)), np.nan, dtype=float)
        W = np.full_like(D, np.nan)

        for i, n in enumerate(n_list):
            if n not in Z_cache:
                Z_cache[n] = precompute_z_big_endian(int(n))
            Z = Z_cache[n]

            for j, p_edge in enumerate(p_list):
                deltas = []
                wins = 0
                used = 0

                for s in seeds:
                    rng = np.random.default_rng(to_uint_seed(s))

                    # generate graph (retry a bit if empty)
                    edges = generate_graph(gc, int(n), float(p_edge), rng, ws_beta=float(a.ws_beta))
                    retry = 0
                    while (not edges) and retry < 50:
                        retry += 1
                        edges = generate_graph(gc, int(n), float(p_edge), rng, ws_beta=float(a.ws_beta))

                    if not edges:
                        continue

                    cut_mask = build_cut_mask(edges, Z)
                    fam = Family1D(len(edges), a.kind, (a.lam_min, a.lam_max), rng, K=int(a.periodic_K))
                    J_star, lam_star = classical_Jstar(fam, cut_mask, int(a.lam_grid))

                    # ID run
                    hist_id = run_outer_vqe_id_budgeted(
                        int(n), cut_mask, fam,
                        lam0=float(a.lam0),
                        outer_max=int(a.outer_max),
                        inner=int(a.inner),
                        budget_evals=B,
                        eta0=float(a.eta0),
                        eta_pow=float(a.eta_pow),
                        step_clip=float(a.step_clip),
                        seed=int(s) + 0,
                        L_vqe=int(a.L_vqe),
                    )

                    # BB-FD run
                    hist_fd = run_outer_vqe_bbfd_budgeted(
                        int(n), cut_mask, fam,
                        lam0=float(a.lam0),
                        outer_max=int(a.outer_max),
                        inner=int(a.inner),
                        budget_evals=B,
                        eta0=float(a.eta0),
                        eta_pow=float(a.eta_pow),
                        step_clip=float(a.step_clip),
                        seed=int(s) + 100000,
                        L_vqe=int(a.L_vqe),
                        fd_c_frac=float(a.fd_c_frac),
                    )

                    # normalize best curves
                    y_id = np.clip(hist_id["J_best"] / J_star, 0.0, 1.5)
                    y_fd = np.clip(hist_fd["J_best"] / J_star, 0.0, 1.5)

                    auc_id = step_auc(hist_id["evals_cum"], y_id, B) / B
                    auc_fd = step_auc(hist_fd["evals_cum"], y_fd, B) / B
                    delta = float(auc_id - auc_fd)

                    deltas.append(delta)
                    wins += int(delta > 1e-12)
                    used += 1

                    # also store final-at-B (step function is already best-so-far)
                    # we approximate by taking last recorded <= B
                    def _value_at(evals_arr, y_arr, x):
                        evals_arr = np.asarray(evals_arr, float)
                        y_arr = np.asarray(y_arr, float)
                        if evals_arr.size == 0:
                            return 0.0
                        idx = np.searchsorted(evals_arr, float(x), side="right") - 1
                        if idx < 0:
                            return 0.0
                        return float(y_arr[idx])

                    final_id = _value_at(hist_id["evals_cum"], y_id, B)
                    final_fd = _value_at(hist_fd["evals_cum"], y_fd, B)

                    rows.append({
                        "graph_class": gc,
                        "n": int(n),
                        "p": float(p_edge),
                        "seed": int(s),
                        "m_edges": int(len(edges)),
                        "kind": str(a.kind),
                        "periodic_K": int(a.periodic_K),
                        "lam0": float(a.lam0),
                        "lam_star_grid": float(lam_star),
                        "J_star_grid": float(J_star),
                        "budget_evals": float(B),
                        "inner": int(a.inner),
                        "outer_max": int(a.outer_max),
                        "L_vqe": int(a.L_vqe),
                        "eta0": float(a.eta0),
                        "eta_pow": float(a.eta_pow),
                        "step_clip": float(a.step_clip),
                        "fd_c_frac": float(a.fd_c_frac),
                        "auc_id": float(auc_id),
                        "auc_fd": float(auc_fd),
                        "delta_auc": float(delta),
                        "final_id": float(final_id),
                        "final_fd": float(final_fd),
                        "delta_final": float(final_id - final_fd),
                    })

                if used > 0:
                    D[i, j] = float(np.mean(deltas))
                    W[i, j] = float(wins / used)
                    agg_rows.append({
                        "graph_class": gc,
                        "n": int(n),
                        "p": float(p_edge),
                        "seeds_used": int(used),
                        "delta_auc_mean": float(np.mean(deltas)),
                        "delta_auc_std": float(np.std(deltas, ddof=1)) if used > 1 else 0.0,
                        "win_rate": float(wins / used),
                    })

                print(f"[{_pretty_graph_name(gc):>14}] n={n:2d} p={p_edge:.2f} | seeds={used:2d} | ΔAUC={D[i,j]:+.4f} | win={100*W[i,j]:.0f}%")

        delta_mats[gc] = D
        win_mats[gc] = W

    # write outputs
    write_csv(out / "exp2D_rows.csv", rows)
    write_csv(out / "exp2D_agg.csv", agg_rows)

    # heatmap
    plot_heatmaps(
        out / f"fig_exp2D_heatmap_deltaAUC.{a.fmt}",
        n_list, p_list, graph_classes,
        delta_mats, win_mats,
        fmt=a.fmt,
        annotate=(not a.no_annotate),
        cmap=str(a.cmap),
    )

    # summary text
    lines = []
    lines.append("Experiment 2D: Heatmap over (n,p) showing ΔAUC_B(ID - BB-FD)")
    lines.append(f"Graph classes: {', '.join(graph_classes)}")
    lines.append(f"n_list: {n_list}")
    lines.append(f"p_list: {p_list}")
    lines.append(f"seeds: {seeds} (N={len(seeds)})")
    lines.append("")
    lines.append(f"Budget B (energy evals): {B}")
    lines.append(f"Inner iters: {a.inner} | Outer max: {a.outer_max}")
    lines.append(f"VQE depth L: {a.L_vqe}")
    lines.append(f"Family: kind={a.kind} periodic_K={a.periodic_K}")
    lines.append(f"FD step: fd_c_frac={a.fd_c_frac}")
    lines.append("")
    lines.append("Legend: cell text shows")
    lines.append("  line1: mean ΔAUC_B (ID - BB-FD)")
    lines.append("  line2: win-rate % (fraction seeds with ΔAUC_B>0)")
    lines.append("")
    lines.append("Saved outputs:")
    lines.append(f"  - exp2D_rows.csv")
    lines.append(f"  - exp2D_agg.csv")
    lines.append(f"  - fig_exp2D_heatmap_deltaAUC.{a.fmt}")
    lines.append(f"  - exp2D_summary.txt")

    with open(out / "exp2D_summary.txt", "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print("Saved to:", out.resolve())


if __name__ == "__main__":
    main()
