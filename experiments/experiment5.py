#!/usr/bin/env python3
"""Deprecated wrapper for exp05_inner_budget_ablation.py.

Deprecated in v0.2.x; remove in v0.3.0.
"""

from __future__ import annotations

import warnings

from exp05_inner_budget_ablation import main

if __name__ == "__main__":
    warnings.warn(
        "'experiments/experiment5.py' is deprecated and will be removed in v0.3.0; use 'experiments/exp05_inner_budget_ablation.py' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    main()
