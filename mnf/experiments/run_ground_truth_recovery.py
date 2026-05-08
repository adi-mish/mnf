from __future__ import annotations

import json
import numpy as np
from mnf.benchmarks.synthetic import sample_chain_data, make_chain_program
from mnf.core.metrics import graph_f1
from mnf.core.validation import prune_edges_by_intervention
from mnf.discovery.graph_discovery import discover_linear_effect_graph


def run(seed: int = 0) -> dict[str, object]:
    inputs, states_list = sample_chain_data(n=512, seed=seed)
    states = {k: np.array([s[k] for s in states_list]) for k in ["x", "a", "b", "y"]}
    pred_edges = discover_linear_effect_graph(states, threshold=0.15)
    true_prog = make_chain_program()
    pruned_edges = prune_edges_by_intervention(true_prog, inputs, pred_edges)
    f1 = graph_f1(true_prog.edges, pred_edges)
    pruned_f1 = graph_f1(true_prog.edges, pruned_edges)
    return {
        "true_edges": sorted(true_prog.edges),
        "pred_edges": sorted(pred_edges),
        "intervention_pruned_edges": sorted(pruned_edges),
        "precision": f1.precision,
        "recall": f1.recall,
        "f1": f1.f1,
        "pruned_precision": pruned_f1.precision,
        "pruned_recall": pruned_f1.recall,
        "pruned_f1": pruned_f1.f1,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
