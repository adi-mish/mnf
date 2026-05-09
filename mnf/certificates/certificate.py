from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from mnf.certificates.identification import IdentificationSet, point_identified


_COST_FIELDS = (
    "obs_error",
    "intervention_error",
    "invariance_error",
    "glue_error",
    "naturalness_cost",
    "closure_error",
    "shared_description_length",
    "uncertainty",
    "identification_diameter",
)


@dataclass(frozen=True)
class ConfidenceInterval:
    low: float
    high: float

    def __post_init__(self) -> None:
        if self.high < self.low:
            raise ValueError("confidence interval high must be >= low")

    @property
    def width(self) -> float:
        return float(self.high - self.low)

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high

    def as_dict(self) -> dict[str, float]:
        return {"low": float(self.low), "high": float(self.high), "width": self.width}


@dataclass(frozen=True)
class MechanismCertificate:
    """Vector certificate for accepting or rejecting a mechanism claim.

    Error, cost, length, and uncertainty dimensions are minimized. `effect` is
    maximized. Scalar scores are exposed only as a configurable Lagrangian for
    search and ranking, not as the canonical truth predicate.
    """

    obs_error: float = 0.0
    intervention_error: float = 0.0
    invariance_error: float = 0.0
    glue_error: float = 0.0
    naturalness_cost: float = 0.0
    closure_error: float = 0.0
    shared_description_length: float = 0.0
    effect: float = 0.0
    uncertainty: float = 0.0
    identification: IdentificationSet = field(default_factory=point_identified)
    confidence_intervals: Mapping[str, ConfidenceInterval] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in self.as_dict(include_intervals=False).items():
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")

    @property
    def identification_diameter(self) -> float:
        return float(self.identification.diameter)

    def as_dict(self, include_intervals: bool = True) -> dict[str, object]:
        out: dict[str, object] = {
            "obs_error": float(self.obs_error),
            "intervention_error": float(self.intervention_error),
            "invariance_error": float(self.invariance_error),
            "glue_error": float(self.glue_error),
            "naturalness_cost": float(self.naturalness_cost),
            "closure_error": float(self.closure_error),
            "shared_description_length": float(self.shared_description_length),
            "effect": float(self.effect),
            "uncertainty": float(self.uncertainty),
            "identification_diameter": float(self.identification.diameter),
        }
        if include_intervals:
            out["confidence_intervals"] = {
                name: interval.as_dict()
                for name, interval in self.confidence_intervals.items()
            }
            out["identification"] = self.identification.as_dict()
        return out

    def cost_vector(self) -> dict[str, float]:
        out = {name: float(getattr(self, name)) for name in _COST_FIELDS}
        out["negative_effect"] = -float(self.effect)
        return out

    def to_lagrangian(self, weights: Mapping[str, float] | None = None) -> float:
        weights = weights or {}
        total = 0.0
        for name in _COST_FIELDS:
            total += float(weights.get(name, 1.0)) * float(getattr(self, name))
        total -= float(weights.get("effect", 1.0)) * float(self.effect)
        return float(total)

    def dominates(
        self,
        other: "MechanismCertificate",
        tolerances: Mapping[str, float] | None = None,
        require_strict: bool = True,
    ) -> bool:
        tolerances = tolerances or {}
        better_or_equal = True
        strictly_better = False
        for name in _COST_FIELDS:
            tol = float(tolerances.get(name, 0.0))
            a = float(getattr(self, name))
            b = float(getattr(other, name))
            if a > b + tol:
                better_or_equal = False
                break
            if a < b - tol:
                strictly_better = True
        if better_or_equal:
            tol = float(tolerances.get("effect", 0.0))
            if self.effect + tol < other.effect:
                better_or_equal = False
            elif self.effect > other.effect + tol:
                strictly_better = True
        return bool(better_or_equal and (strictly_better or not require_strict))
