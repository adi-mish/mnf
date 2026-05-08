from __future__ import annotations

from mnf.interactions.factorial_effects import classify_pairwise_interaction, factorial_from_callable
from mnf.interactions.redundancy import single_ablation_misses_pair


def redundant_or_behavior(left_path_on: bool, right_path_on: bool) -> float:
    return float(left_path_on or right_path_on)


def redundant_paths_metrics() -> dict[str, float | bool | str]:
    effects = factorial_from_callable(redundant_or_behavior)
    out: dict[str, float | bool | str] = effects.as_dict()
    out["interaction_label"] = classify_pairwise_interaction(effects)
    out["single_ablation_misses_pair"] = single_ablation_misses_pair(effects)
    return out
