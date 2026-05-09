from __future__ import annotations

from collections.abc import Sequence

from mnf.models import run_redundant_cpu_sweep


def run(
    seeds: Sequence[int] = (0, 1, 2),
    modulus: int = 7,
    steps: int = 160,
) -> dict[str, object]:
    return run_redundant_cpu_sweep(seeds=seeds, modulus=modulus, steps=steps)
