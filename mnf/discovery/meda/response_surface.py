from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from mnf.interactions.design import (
    InterventionState,
    evaluate_design,
    full_factorial_design,
    pairwise_effects_from_observations,
    pairwise_factorial_design,
    sparse_higher_order_design,
)
from mnf.interactions.factorial_effects import FactorialEffects
from mnf.interactions.higher_order import HigherOrderContrast, search_higher_order_interactions


@dataclass(frozen=True)
class ResponseSurface:
    """Finite intervention response surface for a named mechanism set."""

    names: tuple[str, ...]
    observations: dict[tuple[int, ...], float]
    design: tuple[InterventionState, ...]
    context_on: bool = True

    @classmethod
    def from_callable(
        cls,
        names: Sequence[str],
        behavior_fn: Callable[[Mapping[str, bool]], float],
        design: Sequence[InterventionState] | None = None,
        max_order: int = 2,
        full_factorial: bool = False,
        context_on: bool = True,
    ) -> "ResponseSurface":
        names_tuple = tuple(names)
        if design is None:
            if full_factorial:
                design = full_factorial_design(len(names_tuple))
            elif max_order <= 2:
                design = pairwise_factorial_design(names_tuple, context_on=context_on)
            else:
                design = sparse_higher_order_design(names_tuple, max_order=max_order, context_on=context_on)
        design_tuple = tuple(design)
        observations = evaluate_design(names_tuple, behavior_fn, design_tuple)
        return cls(names=names_tuple, observations=observations, design=design_tuple, context_on=context_on)

    def pairwise_factorials(self) -> dict[tuple[str, str], FactorialEffects]:
        return pairwise_effects_from_observations(self.names, self.observations, context_on=self.context_on)

    def higher_order_contrasts(
        self,
        min_order: int = 3,
        max_order: int | None = None,
        tolerance: float = 1e-7,
    ) -> tuple[HigherOrderContrast, ...]:
        return search_higher_order_interactions(
            self.names,
            self.observations,
            min_order=min_order,
            max_order=max_order,
            tolerance=tolerance,
            context_on=self.context_on,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "names": list(self.names),
            "context_on": self.context_on,
            "design_size": len(self.design),
            "observations": {"".join(map(str, state)): value for state, value in self.observations.items()},
        }
