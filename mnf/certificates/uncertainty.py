from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from mnf.certificates.certificate import ConfidenceInterval


def percentile_interval(
    values: Sequence[float],
    confidence: float = 0.95,
) -> ConfidenceInterval:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("values must not be empty")
    alpha = (1.0 - confidence) / 2.0
    return ConfidenceInterval(
        low=float(np.quantile(arr, alpha)),
        high=float(np.quantile(arr, 1.0 - alpha)),
    )


def mean_interval_width(intervals: Sequence[ConfidenceInterval]) -> float:
    if not intervals:
        return 0.0
    return float(np.mean([interval.width for interval in intervals]))
