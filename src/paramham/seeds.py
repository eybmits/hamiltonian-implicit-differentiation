"""Reproducible seed conversion utilities."""


def to_uint_seed(seed: int) -> int:
    """Convert any integer seed to a valid uint32 seed for numpy."""
    return int(seed) % (2**32 - 1)
