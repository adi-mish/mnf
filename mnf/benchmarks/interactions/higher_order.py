from __future__ import annotations

from collections.abc import Mapping

from mnf.interactions import evaluate_design, higher_order_effect_from_observations, sparse_higher_order_design


def triple_gate_behavior(state: Mapping[str, bool]) -> float:
    return float(state["gate_a"] and state["gate_b"] and state["worker"])


def higher_order_interaction_metrics() -> dict[str, float | int]:
    names = ("gate_a", "gate_b", "worker")
    pairwise_design = sparse_higher_order_design(names, max_order=2)
    triple_design = sparse_higher_order_design(names, max_order=3)
    triple_observations = evaluate_design(names, triple_gate_behavior, triple_design)
    return {
        "pairwise_design_size": len(pairwise_design),
        "triple_design_size": len(triple_design),
        "third_order_effect": higher_order_effect_from_observations(names, triple_observations, names),
    }
