from mnf.benchmarks.interactions.table_aliasing import (
    directed_gate_internal_evidence,
    table_aliasing_metrics,
)
from mnf.interactions import factorial_from_callable, structural_interaction_evidence


def test_same_output_table_is_structurally_ambiguous_without_internal_evidence():
    out = table_aliasing_metrics()

    assert out["directed_equals_symmetric_table"] is True
    assert out["behavior_only"]["structural_label"] == "ambiguous_without_internal_evidence"
    assert out["behavior_only"]["is_ambiguous"] is True


def test_internal_evidence_orients_directed_gate():
    gate_to_worker, worker_to_gate = directed_gate_internal_evidence()
    evidence = structural_interaction_evidence(
        factorial_from_callable(lambda gate, worker: float(gate and worker)),
        internal_gate_m_to_n=gate_to_worker,
        internal_gate_n_to_m=worker_to_gate,
    )

    assert evidence.structural_label == "directed_gate"
    assert evidence.orientation == "m_to_n"
    assert gate_to_worker > 0.0
    assert worker_to_gate == 0.0


def test_saturation_is_distinct_from_synergy_table():
    out = table_aliasing_metrics()

    assert out["saturation_table"]["synergy"] < 0.0
    assert out["saturation_table"] != out["directed_table"]
