from __future__ import annotations

from collections.abc import Mapping

from mnf.semantics.response_kernel import ResponseKernel, kernel_max_distance


def indistinguishable_representatives(
    kernels: Mapping[str, ResponseKernel],
    reference: str,
    epsilon: float = 1e-8,
) -> tuple[str, ...]:
    if reference not in kernels:
        raise ValueError(f"unknown reference kernel: {reference}")
    ref = kernels[reference]
    return tuple(
        name
        for name, kernel in kernels.items()
        if kernel_max_distance(ref, kernel) <= epsilon
    )
