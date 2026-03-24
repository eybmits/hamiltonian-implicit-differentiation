"""Shared experiment defaults, paths, budget costs, and deterministic instance helpers."""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

from paramham.families import Family1D
from paramham.graphs import generate_er_graph, generate_graph
from paramham.seeds import to_uint_seed


@dataclass(frozen=True)
class CanonicalExperimentSetup:
    """Repo-wide default setup for the final experiment reruns."""

    budget_evals: float = 2000.0
    family: str = "periodic"
    periodic_K: int = 6
    n: int = 12
    p_edge: float = 0.45
    lam_min: float = -5.0
    lam_max: float = 5.0
    lam0: float = 0.8
    graph_seed: int = 7


CANONICAL_SETUP = CanonicalExperimentSetup()


def publication_output_dir(experiment_id: str, *parts: str) -> Path:
    """Return the canonical checked-in output directory for one experiment."""

    path = Path("output") / str(experiment_id)
    for part in parts:
        path /= str(part)
    return path


def publication_cache_dir(experiment_id: str, *parts: str) -> Path:
    """Return the canonical checked-in cache directory for one experiment."""

    path = Path("output") / "cache" / str(experiment_id)
    for part in parts:
        path /= str(part)
    return path


def derive_seed(base_seed: int, *parts: Any) -> int:
    """Derive a deterministic child seed from a base seed and stable labels."""

    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return to_uint_seed(int(base_seed) + zlib.crc32(payload))


def _instance_rng(
    *,
    base_seed: int,
    graph_class: str,
    n: int,
    p: float,
    instance_id: Any = None,
    ws_beta: float = 0.3,
    retry: int = 0,
) -> np.random.Generator:
    """Return the deterministic RNG stream for one experiment instance.

    If ``instance_id`` is omitted, the first draw uses the raw seed stream.
    This preserves the legacy single-instance behavior where the graph and the
    family were sampled from one shared RNG seeded directly by ``seed``.
    """

    if instance_id is None and int(retry) == 0:
        return np.random.default_rng(to_uint_seed(int(base_seed)))

    return np.random.default_rng(
        derive_seed(
            int(base_seed),
            str(graph_class).strip().lower(),
            int(n),
            f"{float(p):.8f}",
            f"{float(ws_beta):.8f}",
            instance_id,
            int(retry),
        )
    )


def generate_er_instance_graph(
    n: int,
    p_edge: float,
    *,
    graph_seed: int,
    instance_id: Any = 0,
    retries: int = 50,
) -> List[Tuple[int, int]]:
    """Generate a deterministic non-empty ER graph for one experiment instance."""

    for retry in range(int(retries) + 1):
        rng = _instance_rng(
            base_seed=graph_seed,
            graph_class="er",
            n=int(n),
            p=float(p_edge),
            instance_id=instance_id,
            retry=retry,
        )
        edges = generate_er_graph(int(n), float(p_edge), rng)
        if edges:
            return edges
    return []


def generate_instance_graph(
    graph_class: str,
    n: int,
    p: float,
    *,
    graph_seed: int,
    instance_id: Any = 0,
    ws_beta: float = 0.3,
    retries: int = 50,
) -> List[Tuple[int, int]]:
    """Generate a deterministic non-empty graph from the shared graph dispatcher."""

    for retry in range(int(retries) + 1):
        rng = _instance_rng(
            base_seed=graph_seed,
            graph_class=str(graph_class).strip().lower(),
            n=int(n),
            p=float(p),
            instance_id=instance_id,
            ws_beta=float(ws_beta),
            retry=retry,
        )
        edges = generate_graph(str(graph_class), int(n), float(p), rng, ws_beta=float(ws_beta))
        if edges:
            return edges
    return []


def generate_family1d_instance(
    graph_class: str,
    n: int,
    p: float,
    kind: str,
    lam_bounds: Tuple[float, float],
    *,
    graph_seed: int,
    periodic_K: int = 6,
    instance_id: Any = None,
    ws_beta: float = 0.3,
    retries: int = 50,
    safety_bounds: bool = False,
) -> Tuple[List[Tuple[int, int]], Optional[Family1D]]:
    """Generate a deterministic graph + Family1D pair from one RNG stream."""

    graph_class = str(graph_class).strip().lower()
    for retry in range(int(retries) + 1):
        rng = _instance_rng(
            base_seed=graph_seed,
            graph_class=graph_class,
            n=int(n),
            p=float(p),
            instance_id=instance_id,
            ws_beta=float(ws_beta),
            retry=retry,
        )
        edges = generate_graph(graph_class, int(n), float(p), rng, ws_beta=float(ws_beta))
        if not edges:
            continue
        fam = Family1D(
            len(edges),
            str(kind),
            tuple(map(float, lam_bounds)),
            rng,
            periodic_K=int(periodic_K),
            safety_bounds=bool(safety_bounds),
        )
        return edges, fam
    return [], None


def generate_er_family1d_instance(
    n: int,
    p_edge: float,
    kind: str,
    lam_bounds: Tuple[float, float],
    *,
    graph_seed: int,
    periodic_K: int = 6,
    instance_id: Any = None,
    retries: int = 50,
    safety_bounds: bool = False,
) -> Tuple[List[Tuple[int, int]], Optional[Family1D]]:
    """Generate a deterministic ER graph + Family1D pair from one RNG stream."""

    return generate_family1d_instance(
        "er",
        int(n),
        float(p_edge),
        str(kind),
        tuple(map(float, lam_bounds)),
        graph_seed=int(graph_seed),
        periodic_K=int(periodic_K),
        instance_id=instance_id,
        retries=int(retries),
        safety_bounds=bool(safety_bounds),
    )


def spsa_inner_eval_cost(inner_iters: int) -> int:
    """Cost of one SPSA inner solve in energy evaluations."""

    return 3 * int(inner_iters)


def value_eval_cost(inner_iters: int) -> int:
    """Cost of one center value query: inner solve plus one final expectation eval."""

    return spsa_inner_eval_cost(inner_iters) + 1


def restarted_value_eval_cost(inner_iters: int, restarts: int) -> int:
    """Cost of one value query when the inner solve uses multiple restarts."""

    return int(restarts) * spsa_inner_eval_cost(inner_iters) + 1


def vqe_id_step_cost(inner_iters: int) -> int:
    """Cost of one VQE+ID outer step."""

    return value_eval_cost(inner_iters)


def vqe_fd_value_step_cost(inner_iters: int) -> int:
    """Cost of one VQE+FD outer step with center/+/- value queries."""

    return 3 * value_eval_cost(inner_iters)


def restarted_id_step_cost(inner_iters: int, restarts: int) -> int:
    """Cost of one restarted ID outer step."""

    return restarted_value_eval_cost(inner_iters, restarts)


def restarted_fd_step_cost(inner_iters: int, restarts: int) -> int:
    """Cost of one restarted FD outer step with center/+/- value queries."""

    return 3 * restarted_value_eval_cost(inner_iters, restarts)


def qaoa_full_step_cost(inner_iters: int) -> int:
    """Cost of one QAOA bridge outer step: center value query plus two FD probes."""

    return value_eval_cost(inner_iters) + 2


def can_run_step(*, evals_used: float, budget_evals: float, step_cost: int) -> bool:
    """Return whether a full outer step still fits inside the requested budget."""

    return float(evals_used) + float(step_cost) <= float(budget_evals) + 1e-12
