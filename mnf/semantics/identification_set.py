from __future__ import annotations

from collections.abc import Mapping, Sequence

from mnf.certificates import IdentificationSet
from mnf.semantics.response_kernel import ResponseKernel, kernel_max_distance


def identification_set_from_kernels(
    kernels: Mapping[str, ResponseKernel],
    reference: str,
    epsilon: float = 1e-8,
    diameter: float | None = None,
    reason: str = "response_kernel_indistinguishable",
    distinguishable_by: Sequence[str] = (),
    indistinguishable_under: Sequence[str] = (),
) -> IdentificationSet:
    if reference not in kernels:
        raise ValueError(f"unknown reference kernel: {reference}")
    ref = kernels[reference]
    distances = {name: kernel_max_distance(ref, kernel) for name, kernel in kernels.items()}
    representatives = tuple(name for name, distance in distances.items() if distance <= epsilon)
    id_diameter = max((distances[name] for name in representatives), default=0.0) if diameter is None else diameter
    return IdentificationSet(
        representatives=representatives,
        diameter=float(id_diameter),
        reason=reason,
        distinguishable_by=tuple(distinguishable_by),
        indistinguishable_under=tuple(indistinguishable_under),
        metadata={"distances": {name: float(value) for name, value in distances.items()}},
    )
