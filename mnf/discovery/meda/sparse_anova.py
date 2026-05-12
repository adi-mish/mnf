from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations, product


def boolean_fourier_coefficients(
    names: Sequence[str],
    observations: Mapping[tuple[int, ...], float],
    max_order: int | None = None,
) -> dict[tuple[str, ...], float]:
    """Walsh/Fourier coefficients on a complete {0,1} response table."""

    names = tuple(names)
    n = len(names)
    max_order = n if max_order is None else min(max_order, n)
    expected = set(product((0, 1), repeat=n))
    missing = expected - set(observations)
    if missing:
        raise ValueError(f"complete factorial table required; missing {len(missing)} cells")
    coeffs: dict[tuple[str, ...], float] = {}
    for order in range(max_order + 1):
        for idxs in combinations(range(n), order):
            total = 0.0
            for bits in expected:
                parity = sum(bits[idx] for idx in idxs)
                total += ((-1.0) ** parity) * float(observations[tuple(bits)])
            coeffs[tuple(names[idx] for idx in idxs)] = float(total / (2**n))
    return coeffs


def sparse_anova_terms(
    names: Sequence[str],
    observations: Mapping[tuple[int, ...], float],
    max_order: int | None = None,
    threshold: float = 1e-9,
) -> dict[tuple[str, ...], float]:
    coeffs = boolean_fourier_coefficients(names, observations, max_order=max_order)
    return {term: value for term, value in coeffs.items() if abs(value) > threshold}
