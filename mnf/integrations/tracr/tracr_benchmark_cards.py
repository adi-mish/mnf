from __future__ import annotations

from mnf.benchmarks.cards import BenchmarkCard


def tracr_benchmark_cards() -> tuple[BenchmarkCard, ...]:
    return (
        BenchmarkCard(
            name="tracr_interacting_programs",
            ground_truth_atoms=("compiled_rasp_variable", "attention_pattern", "mlp_transform"),
            ground_truth_mechanisms=("shared_subroutine", "context_gate", "redundant_subroutine"),
            ground_truth_interactions=("shared_reuse", "gating", "redundancy"),
            ground_truth_charts=("compiled_program_chart",),
            allowed_interventions=("activation_patch", "path_patch", "compiled_variable_patch"),
            observable_variables=("logits", "residual_stream", "compiled_rasp_states"),
            known_nonidentifiabilities=("not executable until Tracr is installed",),
            baseline_failure_modes=("component-level circuit discovery misses interacting compiled subroutines",),
            expected_certificate_thresholds={"max_intervention_error": 0.05},
        ),
    )
