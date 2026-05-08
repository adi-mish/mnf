from __future__ import annotations

import json
from itertools import combinations, product
from typing import Any, Sequence

from mnf.benchmarks.gated import exhaustive_gated_xor_inputs, make_gated_xor_program
from mnf.benchmarks.modular import make_modular_addition_program
from mnf.benchmarks.relational import RELATIONS, SUBJECTS, make_relational_lookup_program
from mnf.benchmarks.synthetic import make_chain_program, sample_chain_data
from mnf.core.causal_program import CausalProgram
from mnf.core.metrics import graph_f1
from mnf.core.validation import prune_edges_by_intervention


def _all_ordered_candidates(program: CausalProgram) -> set[tuple[str, str]]:
    return {(src, dst) for src, dst in combinations(program.node_names, 2)}


def _score_program(
    name: str,
    program: CausalProgram,
    inputs: Sequence[dict[str, Any]],
) -> dict[str, object]:
    candidate_edges = _all_ordered_candidates(program)
    pred_edges = prune_edges_by_intervention(program, inputs, candidate_edges)
    f1 = graph_f1(program.edges, pred_edges)
    return {
        "benchmark": name,
        "true_edges": sorted(program.edges),
        "candidate_edges": sorted(candidate_edges),
        "pred_edges": sorted(pred_edges),
        "precision": f1.precision,
        "recall": f1.recall,
        "f1": f1.f1,
    }


def run(seed: int = 0) -> dict[str, object]:
    chain_program = make_chain_program()
    chain_inputs, _chain_states = sample_chain_data(256, seed=seed)

    gated_program = make_gated_xor_program()
    gated_inputs = exhaustive_gated_xor_inputs()

    modular_program = make_modular_addition_program(period=7)
    modular_inputs = [{"a": a, "b": b} for a, b in product(range(7), repeat=2)]

    relational_program = make_relational_lookup_program()
    relational_inputs = [
        {"subject": subject, "relation": relation}
        for subject, relation in product(SUBJECTS, RELATIONS)
    ]

    rows = [
        _score_program("scalar_chain", chain_program, chain_inputs),
        _score_program("gated_xor", gated_program, gated_inputs),
        _score_program("modular_addition_C7", modular_program, modular_inputs),
        _score_program("relational_lookup", relational_program, relational_inputs),
    ]
    return {"rows": rows}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
