from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations, product

from mnf.interactions.design import higher_order_effect_from_observations


@dataclass(frozen=True)
class HigherOrderContrast:
    subset: tuple[str, ...]
    order: int
    effect: float | None
    label: str
    tolerance: float
    uncertainty: float
    missing_cells: int

    def as_dict(self) -> dict[str, object]:
        return {
            "subset": list(self.subset),
            "order": self.order,
            "effect": self.effect,
            "label": self.label,
            "tolerance": self.tolerance,
            "uncertainty": self.uncertainty,
            "missing_cells": self.missing_cells,
        }


def required_cells_for_subset(
    names: Sequence[str],
    subset: Sequence[str],
    context_on: bool = True,
) -> tuple[tuple[int, ...], ...]:
    index = {name: i for i, name in enumerate(names)}
    subset_indices = tuple(index[name] for name in subset)
    context_value = 1 if context_on else 0
    cells: list[tuple[int, ...]] = []
    for bits in product((0, 1), repeat=len(subset_indices)):
        values = [context_value] * len(names)
        for idx, bit in zip(subset_indices, bits):
            values[idx] = bit
        cells.append(tuple(values))
    return tuple(cells)


def classify_higher_order_effect(effect: float, tolerance: float, uncertainty: float = 0.0) -> str:
    margin = tolerance + uncertainty
    if abs(effect) <= margin:
        return "uncertain"
    if effect > 0.0:
        return "positive_higher_order"
    return "negative_higher_order"


def higher_order_contrast(
    names: Sequence[str],
    observations: Mapping[tuple[int, ...], float],
    subset: Sequence[str],
    tolerance: float = 1e-7,
    uncertainty: float = 0.0,
    context_on: bool = True,
) -> HigherOrderContrast:
    subset_tuple = tuple(subset)
    cells = required_cells_for_subset(names, subset_tuple, context_on=context_on)
    missing = sum(1 for cell in cells if cell not in observations)
    if missing:
        return HigherOrderContrast(
            subset=subset_tuple,
            order=len(subset_tuple),
            effect=None,
            label="unobserved",
            tolerance=float(tolerance),
            uncertainty=float(uncertainty),
            missing_cells=missing,
        )
    effect = higher_order_effect_from_observations(names, observations, subset_tuple, context_on=context_on)
    return HigherOrderContrast(
        subset=subset_tuple,
        order=len(subset_tuple),
        effect=float(effect),
        label=classify_higher_order_effect(effect, tolerance=tolerance, uncertainty=uncertainty),
        tolerance=float(tolerance),
        uncertainty=float(uncertainty),
        missing_cells=0,
    )


def search_higher_order_interactions(
    names: Sequence[str],
    observations: Mapping[tuple[int, ...], float],
    min_order: int = 3,
    max_order: int | None = None,
    tolerance: float = 1e-7,
    uncertainty_by_subset: Mapping[tuple[str, ...], float] | None = None,
    context_on: bool = True,
) -> tuple[HigherOrderContrast, ...]:
    if min_order < 2:
        raise ValueError("min_order must be at least 2")
    max_order = len(names) if max_order is None else max_order
    if max_order < min_order:
        raise ValueError("max_order must be greater than or equal to min_order")

    uncertainty_by_subset = uncertainty_by_subset or {}
    contrasts: list[HigherOrderContrast] = []
    for order in range(min_order, min(max_order, len(names)) + 1):
        for subset in combinations(names, order):
            uncertainty = float(uncertainty_by_subset.get(tuple(subset), 0.0))
            contrasts.append(
                higher_order_contrast(
                    names,
                    observations,
                    subset,
                    tolerance=tolerance,
                    uncertainty=uncertainty,
                    context_on=context_on,
                )
            )
    return tuple(contrasts)


def summarize_higher_order_search(contrasts: Sequence[HigherOrderContrast]) -> dict[str, object]:
    counts: dict[str, int] = {}
    for contrast in contrasts:
        counts[contrast.label] = counts.get(contrast.label, 0) + 1
    detected = [
        contrast.as_dict()
        for contrast in contrasts
        if contrast.label in {"positive_higher_order", "negative_higher_order"}
    ]
    return {
        "n_contrasts": len(contrasts),
        "label_counts": counts,
        "detected": detected,
    }
