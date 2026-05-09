from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KernelObservation:
    intervention: str
    context: str
    observable: str
    value: float

    def key(self) -> tuple[str, str, str]:
        return (self.intervention, self.context, self.observable)

    def as_dict(self) -> dict[str, object]:
        return {
            "intervention": self.intervention,
            "context": self.context,
            "observable": self.observable,
            "value": float(self.value),
        }


@dataclass(frozen=True)
class ResponseKernel:
    """Finite response-kernel table over interventions, contexts, observables."""

    values: Mapping[tuple[str, str, str], float]

    def __getitem__(self, key: tuple[str, str, str]) -> float:
        return float(self.values[key])

    @property
    def support(self) -> tuple[tuple[str, str, str], ...]:
        return tuple(sorted(self.values))

    def as_dict(self) -> dict[str, object]:
        return {
            f"{intervention}::{context}::{observable}": float(value)
            for (intervention, context, observable), value in sorted(self.values.items())
        }


def estimate_response_kernel(
    interventions: Sequence[str],
    contexts: Sequence[str],
    observables: Sequence[str],
    response_fn: Callable[[str, str, str], float],
) -> ResponseKernel:
    return ResponseKernel(
        {
            (intervention, context, observable): float(response_fn(intervention, context, observable))
            for intervention in interventions
            for context in contexts
            for observable in observables
        }
    )


def _aligned_values(a: ResponseKernel, b: ResponseKernel) -> tuple[np.ndarray, np.ndarray]:
    a_support = set(a.values)
    b_support = set(b.values)
    if a_support != b_support:
        missing_from_a = sorted(b_support - a_support)
        missing_from_b = sorted(a_support - b_support)
        raise ValueError(
            "response kernels must have identical support; "
            f"missing_from_a={missing_from_a}, missing_from_b={missing_from_b}"
        )
    support = sorted(a_support)
    if not support:
        raise ValueError("response kernels have empty support")
    return (
        np.asarray([a.values[key] for key in support], dtype=float),
        np.asarray([b.values[key] for key in support], dtype=float),
    )


def kernel_l2_distance(a: ResponseKernel, b: ResponseKernel) -> float:
    av, bv = _aligned_values(a, b)
    return float(np.sqrt(np.mean((av - bv) ** 2)))


def kernel_max_distance(a: ResponseKernel, b: ResponseKernel) -> float:
    av, bv = _aligned_values(a, b)
    return float(np.max(np.abs(av - bv)))
