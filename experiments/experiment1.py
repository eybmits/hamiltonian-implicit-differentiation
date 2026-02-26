#!/usr/bin/env python3
"""Deprecated wrapper for exp01_id_vs_fd_core_demo.py.

Deprecated in v0.2.x; remove in v0.3.0.
"""

from __future__ import annotations

import warnings

from exp01_id_vs_fd_core_demo import main

if __name__ == "__main__":
    warnings.warn(
        "'experiments/experiment1.py' is deprecated and will be removed in v0.3.0; use 'experiments/exp01_id_vs_fd_core_demo.py' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    main()
