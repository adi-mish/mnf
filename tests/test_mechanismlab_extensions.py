from itertools import combinations

from mnf.benchmarks.gated import (
    exhaustive_gated_xor_inputs,
    gated_xor_intervention_dataset,
    make_gated_xor_program,
)
from mnf.benchmarks.modular import make_modular_addition_program, modular_shift_error
from mnf.benchmarks.relational import make_relational_lookup_program, relation_swap_dataset
from mnf.benchmarks.synthetic import make_chain_program, sample_chain_data
from mnf.benchmarks.transition_only import make_transition_only_dataset, compare_global_linear_vs_molt
from mnf.core.validation import prune_edges_by_intervention, validate_direct_edge_effects


def test_gated_xor_program_and_gate_interventions():
    prog = make_gated_xor_program()
    rows = [prog.run(inp) for inp in exhaustive_gated_xor_inputs()]
    assert [row["y"] for row in rows] == [0, 0, 0, 1, 0, 1, 0, 0]

    inputs, interventions, states = gated_xor_intervention_dataset()
    for inp, intervention, state in zip(inputs, interventions, states):
        expected_gate = 1 - inp["gate"]
        assert intervention["gate"].value == expected_gate
        assert state["y"] == int(expected_gate and (inp["x1"] ^ inp["x2"]))


def test_modular_addition_has_zero_shift_error():
    prog = make_modular_addition_program(period=7)
    assert prog.run({"a": 5, "b": 6})["sum"] == 4
    assert modular_shift_error(period=7, shift=3) == 0.0


def test_relational_lookup_supports_counterfactual_relation_swaps():
    prog = make_relational_lookup_program()
    assert prog.run({"subject": "alice", "relation": "city"})["object"] == "paris"
    assert prog.run({"subject": "alice", "relation": "instrument"})["object"] == "violin"

    _inputs, _interventions, states = relation_swap_dataset(relation="instrument")
    assert [state["object"] for state in states] == ["violin", "piano", "flute", "drums"]


def test_transition_atoms_beat_global_linear_on_transition_only_task():
    ds = make_transition_only_dataset(n=1500, seed=0)
    metrics = compare_global_linear_vs_molt(ds, seed=0)
    assert metrics["molt_mse"] < 0.1 * metrics["global_linear_mse"]


def test_intervention_validation_prunes_transitive_chain_edges():
    prog = make_chain_program()
    inputs, _states = sample_chain_data(64, seed=0)
    candidate_edges = {
        (src, dst)
        for src, dst in combinations(prog.node_names, 2)
    }
    validations = validate_direct_edge_effects(prog, inputs, candidate_edges)
    assert validations[("x", "a")].direct_effect > 0.0
    assert validations[("x", "b")].direct_effect == 0.0

    pruned = prune_edges_by_intervention(prog, inputs, candidate_edges)
    assert pruned == prog.edges
