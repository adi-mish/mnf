from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass


GaugeMap = Callable[[Mapping[str, float]], Mapping[str, float]]


@dataclass(frozen=True)
class GaugeTransform:
    name: str
    transform: GaugeMap

    def apply(self, state: Mapping[str, float]) -> dict[str, float]:
        return {key: float(value) for key, value in self.transform(state).items()}


def identity_gauge(name: str = "identity") -> GaugeTransform:
    return GaugeTransform(name=name, transform=lambda state: dict(state))


def affine_gauge(scale: float = 1.0, offset: float = 0.0, name: str | None = None) -> GaugeTransform:
    label = name or f"affine(scale={scale},offset={offset})"

    def transform(state: Mapping[str, float]) -> dict[str, float]:
        return {key: scale * float(value) + offset for key, value in state.items()}

    return GaugeTransform(name=label, transform=transform)


def rename_gauge(mapping: Mapping[str, str], name: str = "rename") -> GaugeTransform:
    def transform(state: Mapping[str, float]) -> dict[str, float]:
        return {mapping.get(key, key): float(value) for key, value in state.items()}

    return GaugeTransform(name=name, transform=transform)
