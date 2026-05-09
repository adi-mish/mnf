from __future__ import annotations

from mnf.semantics import estimate_response_kernel, identification_set_from_kernels, kernel_max_distance


INTERVENTIONS = ("none", "ablate_route_a", "ablate_route_b", "ablate_both")
CONTEXTS = ("default",)
OBSERVABLES = ("output",)


def split_route_response(intervention: str, context: str, observable: str) -> float:
    del context, observable
    route_a = intervention not in {"ablate_route_a", "ablate_both"}
    route_b = intervention not in {"ablate_route_b", "ablate_both"}
    return float(route_a or route_b)


def merged_route_response(intervention: str, context: str, observable: str) -> float:
    del context, observable
    # The available interventions cannot distinguish whether one route was
    # internally split: each single ablation leaves the abstract route intact.
    if intervention == "ablate_both":
        return 0.0
    return 1.0


def internal_marker_response(intervention: str, context: str, observable: str) -> float:
    del context
    if observable == "output":
        return float(intervention != "ablate_both")
    if observable == "route_a_marker":
        return float(intervention not in {"ablate_route_a", "ablate_both"})
    if observable == "route_b_marker":
        return float(intervention not in {"ablate_route_b", "ablate_both"})
    raise ValueError(f"unknown observable: {observable}")


def atom_splitting_identifiability_metrics() -> dict[str, object]:
    split = estimate_response_kernel(INTERVENTIONS, CONTEXTS, OBSERVABLES, split_route_response)
    merged = estimate_response_kernel(INTERVENTIONS, CONTEXTS, OBSERVABLES, merged_route_response)
    kernels = {"split_routes": split, "merged_route": merged}
    output_only_id = identification_set_from_kernels(
        kernels,
        reference="split_routes",
        epsilon=0.0,
        diameter=1.0,
        indistinguishable_under=("output_only", "route_ablation"),
        distinguishable_by=("route_internal_markers",),
    )

    richer_observables = ("output", "route_a_marker", "route_b_marker")
    split_rich = estimate_response_kernel(INTERVENTIONS, CONTEXTS, richer_observables, internal_marker_response)
    merged_rich = estimate_response_kernel(INTERVENTIONS, CONTEXTS, richer_observables, merged_route_response)
    rich_distance = kernel_max_distance(split_rich, merged_rich)
    return {
        "output_only_max_distance": kernel_max_distance(split, merged),
        "output_only_identification": output_only_id.as_dict(),
        "rich_observable_max_distance": rich_distance,
        "rich_observables_distinguish": bool(rich_distance > 0.0),
    }
