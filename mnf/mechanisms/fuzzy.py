from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math

from mnf.interactions.overlap import soft_jaccard


@dataclass(frozen=True)
class Atom:
    name: str
    kind: str = "state"
    description_length: float = 1.0

    def __post_init__(self) -> None:
        if self.description_length < 0:
            raise ValueError("description_length must be nonnegative")


@dataclass(frozen=True)
class FuzzyMechanism:
    """Graded causal mechanism over a shared atom library."""

    name: str
    atom_membership: Mapping[str, float]
    effect: float
    intervention_error: float = 0.0
    invariance_error: float = 0.0
    naturalness_cost: float = 0.0
    description_length: float = 1.0
    operating_domain: str = "default"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for atom_name, value in self.atom_membership.items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"membership for {atom_name!r} must be in [0, 1]")
        if self.effect < 0:
            raise ValueError("effect must be nonnegative")
        if self.description_length < 0:
            raise ValueError("description_length must be nonnegative")

    def mechanisticity(
        self,
        lambda_int: float = 1.0,
        lambda_inv: float = 1.0,
        lambda_nat: float = 1.0,
        beta: float = 0.05,
    ) -> float:
        penalty = (
            lambda_int * self.intervention_error
            + lambda_inv * self.invariance_error
            + lambda_nat * self.naturalness_cost
            + beta * self.description_length
        )
        return float(self.effect * math.exp(-penalty))

    def atom_overlap(self, other: "FuzzyMechanism") -> float:
        return soft_jaccard(self.atom_membership, other.atom_membership)

    def active_atoms(self, threshold: float = 1e-12) -> set[str]:
        return {name for name, weight in self.atom_membership.items() if float(weight) > threshold}
