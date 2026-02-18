"""Tests for paramham.seeds."""

from paramham.seeds import to_uint_seed


def test_positive_seed():
    assert to_uint_seed(42) == 42


def test_negative_seed():
    result = to_uint_seed(-1)
    assert 0 <= result < 2**32


def test_large_seed():
    result = to_uint_seed(2**32 + 5)
    assert 0 <= result < 2**32


def test_zero():
    assert to_uint_seed(0) == 0


def test_max_valid():
    assert to_uint_seed(2**32 - 2) == 2**32 - 2
