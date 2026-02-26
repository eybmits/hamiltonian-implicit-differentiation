#!/usr/bin/env python3
"""Deprecated wrapper for exp06_edgewise_lambda_vector.py.

Deprecated in v0.2.x; remove in v0.3.0.
"""

from __future__ import annotations

import warnings

from exp06_edgewise_lambda_vector import main

if __name__ == "__main__":
    warnings.warn(
        "'experiments/experiment6.py' is deprecated and will be removed in v0.3.0; use 'experiments/exp06_edgewise_lambda_vector.py' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    main()
