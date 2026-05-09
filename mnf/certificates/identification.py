from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class IdentificationSet:
    """Non-identifiability certificate for a mechanism claim.

    `diameter` is a task-specific upper bound on how far the listed
    representatives can differ while remaining indistinguishable under the
    current intervention/observable set.
    """

    representatives: tuple[str, ...] = ()
    diameter: float = 0.0
    reason: str = ""
    distinguishable_by: tuple[str, ...] = ()
    indistinguishable_under: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.diameter < 0.0:
            raise ValueError("identification-set diameter must be nonnegative")

    @property
    def is_point_identified(self) -> bool:
        return len(self.representatives) <= 1 and self.diameter == 0.0

    @property
    def is_ambiguous(self) -> bool:
        return not self.is_point_identified

    def as_dict(self) -> dict[str, object]:
        return {
            "representatives": list(self.representatives),
            "diameter": float(self.diameter),
            "reason": self.reason,
            "distinguishable_by": list(self.distinguishable_by),
            "indistinguishable_under": list(self.indistinguishable_under),
            "metadata": dict(self.metadata),
            "is_point_identified": self.is_point_identified,
            "is_ambiguous": self.is_ambiguous,
        }


def point_identified(name: str = "identified") -> IdentificationSet:
    return IdentificationSet(representatives=(name,), diameter=0.0, reason="point_identified")
