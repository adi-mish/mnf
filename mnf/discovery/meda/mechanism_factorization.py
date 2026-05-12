from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from mnf.interactions.factorial_effects import FactorialEffects, classify_pairwise_interaction


@dataclass(frozen=True)
class MechanismFactor:
    pair: tuple[str, str]
    label: str
    effect: float

    def as_dict(self) -> dict[str, object]:
        return {"pair": list(self.pair), "label": self.label, "effect": self.effect}


def factorize_pairwise_mechanisms(
    factorials: Mapping[tuple[str, str], FactorialEffects],
    tolerance: float = 1e-6,
) -> tuple[MechanismFactor, ...]:
    factors = []
    for pair, effects in factorials.items():
        label = classify_pairwise_interaction(effects, tol=tolerance)
        effect = max(abs(effects.joint_effect), abs(effects.synergy), effects.redundancy_score, effects.competition_score)
        factors.append(MechanismFactor(pair=pair, label=label, effect=float(effect)))
    return tuple(factors)
