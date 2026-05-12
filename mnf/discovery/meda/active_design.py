from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from mnf.interactions.active import active_interaction_discovery


def active_meda_design(
    names: Sequence[str],
    behavior_fn: Callable[[Mapping[str, bool]], float],
    noise: float = 0.0,
    seed: int = 0,
    max_measurements: int = 256,
) -> dict[str, object]:
    return active_interaction_discovery(
        names,
        behavior_fn,
        noise=noise,
        seed=seed,
        max_measurements=max_measurements,
    )
