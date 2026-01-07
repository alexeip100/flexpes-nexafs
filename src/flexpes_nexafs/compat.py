"""Compatibility helpers.

This module provides small shims for API changes across dependency versions.
"""

from __future__ import annotations

import numpy as np


def trapezoid(y, x=None, dx: float = 1.0, axis: int = -1):
    """Version-safe trapezoidal integration.

    Uses numpy.trapezoid when available (NumPy >= 2.0), otherwise falls back to numpy.trapz.
    """
    fn = getattr(np, "trapezoid", None)
    if fn is None:
        fn = np.trapz
    return fn(y, x=x, dx=dx, axis=axis)


# Backwards-friendly alias for internal use
trapz = trapezoid
