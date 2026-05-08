from __future__ import annotations

import json
from itertools import combinations

from mnf.benchmarks.gated import exhaustive_gated_xor_inputs, make_gated_xor_program
from mnf.benchmarks.modular import modular_shift_error
from mnf.benchmarks.relational import relation_swap_dataset
from mnf.benchmarks.synthetic import make_chain_program, sample_chain_data
from mnf.benchmarks.transition_only import make_transition_only_dataset, compare_global_linear_vs_molt
from mnf.core.validation import prune_edges_by_intervention


def run(seed: int = 0) -> dict[str, object]:
    gated = make_gated_xor_program()
    gated_rows = [gated.run(inp) for inp in exhaustive_gated_xor_inputs()]

    _rel_inputs, _rel_interventions, rel_states = relation_swap_dataset(relation="instrument")

    transition_ds = make_transition_only_dataset(seed=seed)
    transition_metrics = compare_global_linear_vs_molt(transition_ds, seed=seed)

    chain = make_chain_program()
    chain_inputs, _chain_states = sample_chain_data(128, seed=seed)
    candidate_edges = {(src, dst) for src, dst in combinations(chain.node_names, 2)}
    pruned_edges = prune_edges_by_intervention(chain, chain_inputs, candidate_edges)

    return {
        "gated_xor_truth_table": [
            {"x1": row["x1"], "x2": row["x2"], "gate": row["gate"], "xor": row["xor"], "y": row["y"]}
            for row in gated_rows
        ],
        "modular_shift_error": modular_shift_error(period=7, shift=2),
        "relation_swap_objects": [state["object"] for state in rel_states],
        "transition_only": transition_metrics,
        "intervention_pruned_chain_edges": sorted(pruned_edges),
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
