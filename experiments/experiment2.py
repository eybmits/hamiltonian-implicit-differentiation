#!/usr/bin/env python3
"""Deprecated wrapper for exp02_budget_efficiency_multiseed.py.

Deprecated in v0.2.x; remove in v0.3.0.
"""

from __future__ import annotations

import warnings

from exp02_budget_efficiency_multiseed import main

if __name__ == "__main__":
    warnings.warn(
        "'experiments/experiment2.py' is deprecated and will be removed in v0.3.0; use 'experiments/exp02_budget_efficiency_multiseed.py' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    main()
