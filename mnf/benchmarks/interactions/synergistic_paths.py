from __future__ import annotations

from mnf.interactions.factorial_effects import classify_pairwise_interaction, factorial_from_callable
from mnf.interactions.shapley import pairwise_shapley_values


def and_synergy_behavior(left_path_on: bool, right_path_on: bool) -> float:
    return float(left_path_on and right_path_on)


def synergistic_paths_metrics() -> dict[str, float | str]:
    effects = factorial_from_callable(and_synergy_behavior)
    out: dict[str, float | str] = effects.as_dict()
    out["interaction_label"] = classify_pairwise_interaction(effects)
    out.update({f"shapley_{k}": v for k, v in pairwise_shapley_values(effects).items()})
    return out
