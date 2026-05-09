from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NaturalnessProfile:
    locality_cost: float = 0.0
    description_length: float = 0.0
    intervention_realizability_cost: float = 0.0
    instability_cost: float = 0.0
    type_mismatch_cost: float = 0.0
    gauge_overclaim_cost: float = 0.0
    closure_cost: float = 0.0

    def __post_init__(self) -> None:
        for name, value in self.as_dict().items():
            if value < 0.0:
                raise ValueError(f"{name} must be nonnegative")

    @property
    def total_cost(self) -> float:
        return float(sum(self.as_dict().values()))

    def as_dict(self) -> dict[str, float]:
        return {
            "locality_cost": float(self.locality_cost),
            "description_length": float(self.description_length),
            "intervention_realizability_cost": float(self.intervention_realizability_cost),
            "instability_cost": float(self.instability_cost),
            "type_mismatch_cost": float(self.type_mismatch_cost),
            "gauge_overclaim_cost": float(self.gauge_overclaim_cost),
            "closure_cost": float(self.closure_cost),
        }
