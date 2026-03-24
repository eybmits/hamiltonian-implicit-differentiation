"""Tests for shared experiment defaults and deterministic instance helpers."""

import numpy as np

from paramham.experiment_defaults import (
    CANONICAL_SETUP,
    can_run_step,
    generate_er_family1d_instance,
    generate_er_instance_graph,
    qaoa_full_step_cost,
    restarted_fd_step_cost,
    restarted_id_step_cost,
    value_eval_cost,
    vqe_fd_value_step_cost,
    vqe_id_step_cost,
)
from paramham.families import Family1D
from paramham.graphs import generate_random_graph
from paramham.seeds import to_uint_seed


def test_canonical_setup_defaults():
    assert CANONICAL_SETUP.budget_evals == 2000.0
    assert CANONICAL_SETUP.family == "periodic"
    assert CANONICAL_SETUP.periodic_K == 6
    assert CANONICAL_SETUP.graph_seed == 7


def test_generate_er_instance_graph_is_deterministic():
    g1 = generate_er_instance_graph(12, 0.45, graph_seed=7, instance_id="demo")
    g2 = generate_er_instance_graph(12, 0.45, graph_seed=7, instance_id="demo")
    assert g1 == g2
    assert len(g1) > 0


def test_generate_er_family1d_instance_matches_legacy_single_seed_stream():
    rng = np.random.default_rng(to_uint_seed(7))
    legacy_edges = generate_random_graph(12, 0.45, rng)
    legacy_fam = Family1D(len(legacy_edges), "quadratic", (-5.0, 5.0), rng, periodic_K=6, safety_bounds=False)

    edges, fam = generate_er_family1d_instance(
        12,
        0.45,
        "quadratic",
        (-5.0, 5.0),
        graph_seed=7,
        periodic_K=6,
        safety_bounds=False,
    )

    assert fam is not None
    assert edges == legacy_edges
    assert np.allclose(fam.wbar, legacy_fam.wbar)
    assert np.allclose(fam.A, legacy_fam.A)
    assert np.allclose(fam.s, legacy_fam.s)


def test_step_cost_helpers():
    assert value_eval_cost(10) == 31
    assert vqe_id_step_cost(10) == 31
    assert vqe_fd_value_step_cost(10) == 93
    assert restarted_id_step_cost(10, 4) == 121
    assert restarted_fd_step_cost(10, 4) == 363
    assert qaoa_full_step_cost(100) == 303


def test_can_run_step():
    assert can_run_step(evals_used=1869.0, budget_evals=2000.0, step_cost=31)
    assert not can_run_step(evals_used=1870.0, budget_evals=2000.0, step_cost=131)
