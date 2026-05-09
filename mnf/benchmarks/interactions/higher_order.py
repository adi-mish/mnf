from __future__ import annotations

from collections.abc import Mapping, Sequence

from mnf.interactions import (
    evaluate_design,
    higher_order_effect_from_observations,
    search_higher_order_interactions,
    sparse_higher_order_design,
    summarize_higher_order_search,
)


def triple_gate_behavior(state: Mapping[str, bool]) -> float:
    return float(state["gate_a"] and state["gate_b"] and state["worker"])


def k_way_gate_behavior(state: Mapping[str, bool]) -> float:
    return float(all(state.values()))


def k_way_higher_order_sweep(orders: Sequence[int] = (3, 4, 5)) -> dict[str, object]:
    rows = []
    for order in orders:
        names = tuple(f"m{i}" for i in range(order))
        design = sparse_higher_order_design(names, max_order=order)
        observations = evaluate_design(names, k_way_gate_behavior, design)
        effect = higher_order_effect_from_observations(names, observations, names)
        search = search_higher_order_interactions(names, observations, min_order=order, max_order=order)
        rows.append(
            {
                "order": order,
                "design_size": len(design),
                "effect": float(effect),
                "label": search[0].label,
            }
        )
    return {"rows": rows}


def higher_order_interaction_metrics() -> dict[str, object]:
    names = ("gate_a", "gate_b", "worker")
    pairwise_design = sparse_higher_order_design(names, max_order=2)
    triple_design = sparse_higher_order_design(names, max_order=3)
    pairwise_observations = evaluate_design(names, triple_gate_behavior, pairwise_design)
    triple_observations = evaluate_design(names, triple_gate_behavior, triple_design)
    pairwise_search = search_higher_order_interactions(names, pairwise_observations, min_order=3, max_order=3)
    triple_search = search_higher_order_interactions(names, triple_observations, min_order=3, max_order=3)
    return {
        "pairwise_design_size": len(pairwise_design),
        "triple_design_size": len(triple_design),
        "third_order_effect": higher_order_effect_from_observations(names, triple_observations, names),
        "pairwise_only_search": summarize_higher_order_search(pairwise_search),
        "triple_search": summarize_higher_order_search(triple_search),
        "k_way_sweep": k_way_higher_order_sweep(),
    }
