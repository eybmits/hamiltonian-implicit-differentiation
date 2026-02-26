#!/usr/bin/env python3
"""Deprecated wrapper for exp04_robustness_sweep_periodic_k.py.

Deprecated in v0.2.x; remove in v0.3.0.
"""

from __future__ import annotations

import warnings

from exp04_robustness_sweep_periodic_k import main

if __name__ == "__main__":
    warnings.warn(
        "'experiments/experiment4.py' is deprecated and will be removed in v0.3.0; use 'experiments/exp04_robustness_sweep_periodic_k.py' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    main()
