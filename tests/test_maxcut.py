"""Tests for paramham.maxcut."""

import numpy as np

from paramham.families import Family1D
from paramham.maxcut import build_cut_mask, build_ZZ_edges, classical_Jstar, precompute_z


def test_precompute_z_shape():
    Z = precompute_z(3)
    assert Z.shape == (3, 8)


def test_precompute_z_values():
    Z = precompute_z(2)
    # 2 qubits, 4 states: |00>, |01>, |10>, |11>
    # big-endian: qubit 0 is MSB
    assert Z.shape == (2, 4)
    # |00>: Z0=+1, Z1=+1
    assert Z[0, 0] == 1
    assert Z[1, 0] == 1
    # |01>: Z0=+1, Z1=-1
    assert Z[0, 1] == 1
    assert Z[1, 1] == -1
    # |10>: Z0=-1, Z1=+1
    assert Z[0, 2] == -1
    assert Z[1, 2] == 1
    # |11>: Z0=-1, Z1=-1
    assert Z[0, 3] == -1
    assert Z[1, 3] == -1


def test_build_cut_mask():
    Z = precompute_z(2)
    edges = [(0, 1)]
    cm = build_cut_mask(edges, Z)
    assert cm.shape == (4, 1)
    # |00>: same -> cut=0, |01>: diff -> cut=1, |10>: diff -> cut=1, |11>: same -> cut=0
    np.testing.assert_array_almost_equal(cm[:, 0], [0.0, 1.0, 1.0, 0.0])


def test_build_ZZ_edges():
    Z = precompute_z(2)
    edges = [(0, 1)]
    ZZ = build_ZZ_edges(edges, Z)
    assert ZZ.shape == (1, 4)
    # ZZ = Z0*Z1: |00>:+1, |01>:-1, |10>:-1, |11>:+1
    np.testing.assert_array_equal(ZZ[0], [1, -1, -1, 1])


def test_classical_Jstar():
    rng = np.random.default_rng(42)
    edges = [(0, 1), (1, 2)]
    n = 3
    Z = precompute_z(n)
    cm = build_cut_mask(edges, Z)
    fam = Family1D(len(edges), "linear", (0.0, 1.0), rng)
    J_star, lam_star = classical_Jstar(fam, cm, 101)
    assert J_star > 0
    assert 0.0 <= lam_star <= 1.0
