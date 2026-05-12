from __future__ import annotations

from collections.abc import Callable, Mapping


def causal_scrubbing_residual(
    original_behavior: Callable[[Mapping[str, bool]], float],
    scrubbed_behavior: Callable[[Mapping[str, bool]], float],
    state: Mapping[str, bool],
) -> float:
    return float(abs(original_behavior(state) - scrubbed_behavior(state)))


__all__ = ["causal_scrubbing_residual"]
