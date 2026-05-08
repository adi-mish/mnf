from __future__ import annotations

from collections.abc import Sequence

from mnf.interactions.factorial_effects import classify_pairwise_interaction, factorial_from_callable


def mixed_or_and_behavior(redundancy_weight: float, synergy_weight: float):
    """Return a behavior with tunable redundant and synergistic components."""

    def behavior(m_on: bool, n_on: bool) -> float:
        redundant_part = float(m_on or n_on)
        synergistic_part = float(m_on and n_on)
        return float(redundancy_weight * redundant_part + synergy_weight * synergistic_part)

    return behavior


def factorial_interaction_phase_diagram(
    redundancy_weights: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    synergy_weights: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> dict[str, object]:
    rows = []
    for redundancy_weight in redundancy_weights:
        for synergy_weight in synergy_weights:
            effects = factorial_from_callable(mixed_or_and_behavior(redundancy_weight, synergy_weight))
            rows.append(
                {
                    "redundancy_weight": float(redundancy_weight),
                    "synergy_weight": float(synergy_weight),
                    "synergy": effects.synergy,
                    "redundancy_score": effects.redundancy_score,
                    "compensation_score": effects.compensation_score,
                    "interaction_label": classify_pairwise_interaction(effects),
                }
            )
    return {"rows": rows}
