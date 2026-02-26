#!/usr/bin/env python3
"""Deprecated wrapper for exp08_id_vs_fd_np_graphclass_heatmap.py.

Deprecated in v0.2.x; remove in v0.3.0.
"""

from __future__ import annotations

import warnings

from exp08_id_vs_fd_np_graphclass_heatmap import main

if __name__ == "__main__":
    warnings.warn(
        "'experiments/experiment8.py' is deprecated and will be removed in v0.3.0; use 'experiments/exp08_id_vs_fd_np_graphclass_heatmap.py' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    main()
