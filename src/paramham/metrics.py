"""Evaluation metrics: step interpolation, AUC, mean/stderr, win rate, tail probability."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def step_interp(
    evals: np.ndarray,
    values: np.ndarray,
    budgets: np.ndarray,
    anchor_value: float = 0.0,
    sort: bool = False,
) -> np.ndarray:
    """Piecewise-constant (step-function) interpolation.

    Returns ``v(b) = values[last index with evals <= b]``,
    anchored at ``(0, anchor_value)`` if the first eval > 0.

    Parameters
    ----------
    evals : array
        Cumulative evaluation counts (assumed non-decreasing unless *sort*).
    values : array
        Corresponding metric values.
    budgets : array
        Query points.
    anchor_value : float
        Value to use before the first evaluation event.
    sort : bool
        If True, sort *evals*/*values* by eval before interpolation.
    """
    evals = np.asarray(evals, dtype=float)
    values = np.asarray(values, dtype=float)
    budgets = np.asarray(budgets, dtype=float)

    if evals.size == 0:
        return np.full_like(budgets, anchor_value)

    if sort:
        order = np.argsort(evals)
        evals = evals[order]
        values = values[order]

    if evals[0] > 0:
        evals = np.concatenate([[0.0], evals])
        values = np.concatenate([[anchor_value], values])

    idx = np.searchsorted(evals, budgets, side="right") - 1
    idx = np.clip(idx, 0, values.size - 1)
    return values[idx]


def step_auc(evals: np.ndarray, values: np.ndarray, budget: float) -> float:
    """Normalised AUC of a step function on [0, budget].

    The step function holds each value constant between consecutive eval events.
    Returns area / budget (a value in roughly the same range as the metric).
    """
    evals = np.asarray(evals, float)
    values = np.asarray(values, float)
    budget = float(budget)

    if evals.size == 0 or budget <= 0:
        return 0.0

    if evals[0] > 0.0:
        evals = np.insert(evals, 0, 0.0)
        values = np.insert(values, 0, 0.0)

    m = evals <= budget
    evals = evals[m]
    values = values[m]
    if evals.size == 0:
        return 0.0

    if evals[-1] < budget:
        evals = np.append(evals, budget)
        values = np.append(values, values[-1])

    dx = np.diff(evals)
    area = float(np.sum(dx * values[:-1]))
    return area / budget


def mean_stderr(x: np.ndarray, axis: int = 0):
    """Compute mean and standard error along *axis*."""
    x = np.asarray(x, float)
    mu = np.nanmean(x, axis=axis)
    sd = np.nanstd(x, axis=axis, ddof=1)
    n = np.sum(np.isfinite(x), axis=axis)
    se = sd / np.sqrt(np.maximum(1, n))
    return mu, se


def win_rate(gains: np.ndarray) -> float:
    """Fraction of entries where gain > 0."""
    gains = np.asarray(gains, float)
    finite = gains[np.isfinite(gains)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite > 0))


def tail_probability(
    probs: np.ndarray,
    cut_vals: np.ndarray,
    *,
    metric: str,
    J_star: float,
    hit_eps: float = 0.01,
    topk_frac: float = 0.01,
) -> Tuple[float, float]:
    """Compute tail probability for readout quality.

    Parameters
    ----------
    probs : ndarray
        Computational basis probability distribution.
    cut_vals : ndarray
        Cut value for each computational basis state.
    metric : str
        ``"hit"`` for P(cut >= (1-eps)*J*) or ``"topk"`` for quantile-based.
    J_star : float
        Classical optimum for the hit metric.
    hit_eps, topk_frac : float
        Parameters for the respective metric.

    Returns
    -------
    p_tail : float
        Tail probability.
    threshold : float
        The threshold used.
    """
    metric = str(metric).lower().strip()
    if metric == "hit":
        thr = (1.0 - float(hit_eps)) * float(J_star)
    elif metric == "topk":
        q = float(np.clip(1.0 - float(topk_frac), 0.0, 1.0))
        thr = float(np.quantile(cut_vals, q))
    else:
        return float("nan"), float("nan")

    m = cut_vals >= thr
    p = float(np.clip(np.sum(probs[m]), 0.0, 1.0))
    return p, float(thr)
