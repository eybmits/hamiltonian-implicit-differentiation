#!/usr/bin/env python3
"""Deprecated wrapper for exp07_vqe_vs_qaoa_readout_bridge.py.

Deprecated in v0.2.x; remove in v0.3.0.
"""

from __future__ import annotations

import warnings

from exp07_vqe_vs_qaoa_readout_bridge import main

if __name__ == "__main__":
    warnings.warn(
        "'experiments/experiment7.py' is deprecated and will be removed in v0.3.0; use 'experiments/exp07_vqe_vs_qaoa_readout_bridge.py' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    main()
