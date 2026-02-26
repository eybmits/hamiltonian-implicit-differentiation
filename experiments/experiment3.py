#!/usr/bin/env python3
"""Deprecated wrapper for exp03_readout_realism_best_mode.py.

Deprecated in v0.2.x; remove in v0.3.0.
"""

from __future__ import annotations

import warnings

from exp03_readout_realism_best_mode import main

if __name__ == "__main__":
    warnings.warn(
        "'experiments/experiment3.py' is deprecated and will be removed in v0.3.0; use 'experiments/exp03_readout_realism_best_mode.py' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    main()
