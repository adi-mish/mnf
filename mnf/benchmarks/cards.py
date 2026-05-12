from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkCard:
    name: str
    ground_truth_atoms: tuple[str, ...]
    ground_truth_mechanisms: tuple[str, ...]
    ground_truth_interactions: tuple[str, ...]
    ground_truth_charts: tuple[str, ...]
    allowed_interventions: tuple[str, ...]
    observable_variables: tuple[str, ...]
    known_nonidentifiabilities: tuple[str, ...]
    baseline_failure_modes: tuple[str, ...]
    expected_certificate_thresholds: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "ground_truth_atoms": list(self.ground_truth_atoms),
            "ground_truth_mechanisms": list(self.ground_truth_mechanisms),
            "ground_truth_interactions": list(self.ground_truth_interactions),
            "ground_truth_charts": list(self.ground_truth_charts),
            "allowed_interventions": list(self.allowed_interventions),
            "observable_variables": list(self.observable_variables),
            "known_nonidentifiabilities": list(self.known_nonidentifiabilities),
            "baseline_failure_modes": list(self.baseline_failure_modes),
            "expected_certificate_thresholds": dict(self.expected_certificate_thresholds),
        }


BENCHMARK_CARDS: tuple[BenchmarkCard, ...] = (
    BenchmarkCard(
        name="table_aliasing",
        ground_truth_atoms=("m", "n", "internal_gate_trace"),
        ground_truth_mechanisms=("directed_gate", "symmetric_synergy"),
        ground_truth_interactions=("phenomenological_synergy", "structural_aliasing"),
        ground_truth_charts=("binary_pair_chart",),
        allowed_interventions=("pairwise_on_off", "internal_gate_target_patch"),
        observable_variables=("output", "internal_gate_trace"),
        known_nonidentifiabilities=("output-only factorial table cannot orient directed gate",),
        baseline_failure_modes=("behavior-only structural labeling",),
        expected_certificate_thresholds={"max_identification_diameter_with_internal_evidence": 0.0},
    ),
    BenchmarkCard(
        name="atom_splitting_identifiability",
        ground_truth_atoms=("route_a", "route_b", "merged_route"),
        ground_truth_mechanisms=("redundant_route_factor",),
        ground_truth_interactions=("route_redundancy",),
        ground_truth_charts=("output_only_kernel", "marker_augmented_kernel"),
        allowed_interventions=("route_ablation", "internal_route_marker_observation"),
        observable_variables=("output", "route_marker"),
        known_nonidentifiabilities=("split and merged routes are indistinguishable output-only",),
        baseline_failure_modes=("point estimate under weak observables",),
        expected_certificate_thresholds={"min_output_only_identification_diameter": 1.0},
    ),
    BenchmarkCard(
        name="tiny_shared_residual_redundant_transformer",
        ground_truth_atoms=("shared_residual", "route_a_readout", "route_b_readout"),
        ground_truth_mechanisms=("modular_addition_route_a", "modular_addition_route_b"),
        ground_truth_interactions=("redundancy", "shared_mdl_reuse"),
        ground_truth_charts=("C_7_output_chart", "route_readout_chart"),
        allowed_interventions=("route_a_ablation", "route_b_ablation", "dual_route_ablation"),
        observable_variables=("logits", "route_logits", "accuracy"),
        known_nonidentifiabilities=("routes are named readouts, not hidden-route discovery",),
        baseline_failure_modes=("single-ablation necessity underweights redundant routes",),
        expected_certificate_thresholds={"min_dual_ablation_drop": 0.75, "max_false_redundancy_rate": 0.0},
    ),
)


def benchmark_card_index() -> dict[str, BenchmarkCard]:
    return {card.name: card for card in BENCHMARK_CARDS}
