from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class InterventionAlgebra:
    """Finite intervention algebra for response-kernel experiments."""

    interventions: tuple[str, ...]
    identity: str = "none"
    composition: Mapping[tuple[str, str], str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.identity not in self.interventions:
            raise ValueError("identity intervention must be listed")
        unknown = set()
        for (left, right), result in self.composition.items():
            unknown.update({left, right, result} - set(self.interventions))
        if unknown:
            raise ValueError(f"composition references unknown interventions: {sorted(unknown)}")

    def contains(self, intervention: str) -> bool:
        return intervention in self.interventions

    def compose(self, left: str, right: str) -> str:
        if left not in self.interventions or right not in self.interventions:
            raise ValueError(f"unknown intervention pair: {(left, right)}")
        if left == self.identity:
            return right
        if right == self.identity:
            return left
        try:
            return self.composition[(left, right)]
        except KeyError as exc:
            raise ValueError(f"composition is undefined for {(left, right)}") from exc

    @property
    def is_closed(self) -> bool:
        for left in self.interventions:
            for right in self.interventions:
                try:
                    self.compose(left, right)
                except ValueError:
                    return False
        return True

    def as_dict(self) -> dict[str, object]:
        return {
            "interventions": list(self.interventions),
            "identity": self.identity,
            "composition": {f"{left}::{right}": result for (left, right), result in self.composition.items()},
            "is_closed": self.is_closed,
        }
