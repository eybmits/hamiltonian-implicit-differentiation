"""Tests for paramham.graphs."""

import numpy as np
import pytest

from paramham.graphs import (
    generate_barabasi_albert,
    generate_er_graph,
    generate_graph,
    generate_random_graph,
    generate_regular_ring,
    generate_watts_strogatz,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def _check_edges(edges, n):
    """Common edge validity checks."""
    for i, j in edges:
        assert 0 <= i < n
        assert 0 <= j < n
        assert i < j  # canonical ordering


def test_er_graph(rng):
    edges = generate_er_graph(6, 0.5, rng)
    _check_edges(edges, 6)
    assert len(edges) > 0


def test_er_graph_p0(rng):
    edges = generate_er_graph(6, 0.0, rng)
    assert len(edges) == 0


def test_er_graph_p1(rng):
    edges = generate_er_graph(4, 1.0, rng)
    assert len(edges) == 6  # C(4,2) = 6


def test_random_graph_alias(rng):
    assert generate_random_graph is generate_er_graph


def test_regular_ring():
    edges = generate_regular_ring(6, 2)
    _check_edges(edges, 6)
    assert len(edges) == 6  # ring with degree 2


def test_regular_ring_k4():
    edges = generate_regular_ring(6, 4)
    _check_edges(edges, 6)
    assert len(edges) == 12


def test_regular_ring_odd_k():
    edges = generate_regular_ring(6, 3)
    _check_edges(edges, 6)
    # k=3 gets rounded down to k=2
    assert len(edges) == 6


def test_watts_strogatz(rng):
    edges = generate_watts_strogatz(8, 4, 0.3, rng)
    _check_edges(edges, 8)
    assert len(edges) > 0


def test_watts_strogatz_beta0(rng):
    edges_ws = generate_watts_strogatz(6, 2, 0.0, rng)
    edges_ring = generate_regular_ring(6, 2)
    assert edges_ws == edges_ring


def test_barabasi_albert(rng):
    edges = generate_barabasi_albert(8, 2, rng)
    _check_edges(edges, 8)
    assert len(edges) > 0


def test_barabasi_albert_small(rng):
    edges = generate_barabasi_albert(1, 1, rng)
    assert len(edges) == 0


def test_generate_graph_er(rng):
    edges = generate_graph("er", 6, 0.5, rng)
    _check_edges(edges, 6)


def test_generate_graph_ring(rng):
    edges = generate_graph("ring", 6, 0.5, rng)
    _check_edges(edges, 6)


def test_generate_graph_ws(rng):
    edges = generate_graph("ws", 8, 0.5, rng)
    _check_edges(edges, 8)


def test_generate_graph_ba(rng):
    edges = generate_graph("ba", 8, 0.5, rng)
    _check_edges(edges, 8)


def test_generate_graph_unknown(rng):
    with pytest.raises(ValueError, match="Unknown graph_class"):
        generate_graph("xyz", 6, 0.5, rng)
