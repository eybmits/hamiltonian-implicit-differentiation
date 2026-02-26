#!/usr/bin/env python3
"""Deprecated wrapper for exp02_budget_efficiency_t20_variant.py.

Deprecated in v0.2.x; remove in v0.3.0.
"""

from __future__ import annotations

import warnings

from exp02_budget_efficiency_t20_variant import main

if __name__ == "__main__":
    warnings.warn(
        "'experiments/experiment2_plot.py' is deprecated and will be removed in v0.3.0; use 'experiments/exp02_budget_efficiency_t20_variant.py' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    main()
