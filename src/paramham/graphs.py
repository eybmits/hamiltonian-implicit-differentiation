"""Random graph generators for Max-Cut experiments.

Includes Erdos-Renyi, regular ring, Watts-Strogatz, and Barabasi-Albert.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def _edge_key(i: int, j: int) -> Tuple[int, int]:
    return (i, j) if i < j else (j, i)


# ---------------------------------------------------------------------------
# Erdos-Renyi G(n, p)
# ---------------------------------------------------------------------------


def generate_er_graph(n: int, p_edge: float, rng: np.random.Generator) -> List[Tuple[int, int]]:
    """Generate an Erdos-Renyi random graph G(n, p)."""
    edges: list[Tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p_edge:
                edges.append((i, j))
    return edges


# backward-compat alias used by most experiments
generate_random_graph = generate_er_graph


# ---------------------------------------------------------------------------
# Regular ring lattice
# ---------------------------------------------------------------------------


def generate_regular_ring(n: int, k: int) -> List[Tuple[int, int]]:
    """Undirected ring-lattice with degree *k* (k must be even)."""
    k = int(k)
    k = max(0, min(k, n - 1))
    if k % 2 == 1:
        k -= 1
    if k <= 0:
        return []
    half = k // 2
    edges: set[Tuple[int, int]] = set()
    for i in range(n):
        for d in range(1, half + 1):
            j = (i + d) % n
            edges.add(_edge_key(i, j))
    return sorted(edges)


# ---------------------------------------------------------------------------
# Watts-Strogatz
# ---------------------------------------------------------------------------


def generate_watts_strogatz(n: int, k: int, beta: float, rng: np.random.Generator) -> List[Tuple[int, int]]:
    """Watts-Strogatz small-world: ring lattice with random rewiring."""
    beta = float(np.clip(beta, 0.0, 1.0))
    edges = generate_regular_ring(n, k)
    if not edges or beta <= 0:
        return edges

    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)

    new_edges: set[Tuple[int, int]] = set(map(tuple, edges))  # type: ignore[arg-type]

    for i, j in list(edges):
        if rng.random() >= beta:
            continue
        new_edges.discard(_edge_key(i, j))
        adj[i].discard(j)
        adj[j].discard(i)

        candidates = [x for x in range(n) if x != i and x not in adj[i]]
        if not candidates:
            new_edges.add(_edge_key(i, j))
            adj[i].add(j)
            adj[j].add(i)
            continue
        k_new = int(rng.choice(candidates))
        new_edges.add(_edge_key(i, k_new))
        adj[i].add(k_new)
        adj[k_new].add(i)

    return sorted(new_edges)


# ---------------------------------------------------------------------------
# Barabasi-Albert preferential attachment
# ---------------------------------------------------------------------------


def generate_barabasi_albert(n: int, m: int, rng: np.random.Generator) -> List[Tuple[int, int]]:
    """Simple Barabasi-Albert preferential attachment graph."""
    n = int(n)
    m = int(m)
    if n <= 1:
        return []
    m = max(1, min(m, n - 1))

    m0 = min(n, m + 1)
    edges: set[Tuple[int, int]] = set()
    degree = np.zeros(n, dtype=int)

    for i in range(m0):
        for j in range(i + 1, m0):
            edges.add((i, j))
            degree[i] += 1
            degree[j] += 1

    repeated: list[int] = []
    for i in range(m0):
        repeated.extend([i] * degree[i])

    for new in range(m0, n):
        targets: set[int] = set()
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


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def generate_graph(
    graph_class: str, n: int, p: float, rng: np.random.Generator, ws_beta: float = 0.3
) -> List[Tuple[int, int]]:
    """Generate edges for a given graph class.

    *p* is interpreted as:
      - ER: edge probability
      - ring/ws/ba: mapped to expected degree d = p*(n-1)
    """
    graph_class = str(graph_class).strip().lower()
    p = float(np.clip(p, 0.0, 1.0))

    if graph_class in ("er", "erdos", "erdos_renyi", "gnp"):
        return generate_er_graph(n, p, rng)

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
