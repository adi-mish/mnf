from __future__ import annotations

from mnf.benchmarks.interactions.table_aliasing import table_aliasing_metrics


def same_factorial_table_different_structure_metrics() -> dict[str, object]:
    metrics = table_aliasing_metrics()
    return {
        "same_output_table": metrics["directed_equals_symmetric_table"],
        "directed_table": metrics["directed_table"],
        "symmetric_table": metrics["symmetric_table"],
        "behavior_only_structural_label": metrics["behavior_only"]["structural_label"],
        "directed_with_internal_evidence": metrics["directed_structural"],
        "symmetric_with_internal_evidence": metrics["symmetric_structural"],
        "candidate_structures": ["directed_gate", "symmetric_synergy", "output_saturation"],
    }
