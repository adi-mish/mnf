from __future__ import annotations

import numpy as np

from mnf.baselines.features import linear_probe_direction


def linear_das_direction(source: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """CPU-light linear distributed-alignment-search proxy."""

    return linear_probe_direction(source, labels)


__all__ = ["linear_das_direction"]
