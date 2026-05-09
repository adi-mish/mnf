from __future__ import annotations

from collections.abc import Sequence
import json

import numpy as np

from mnf.benchmarks.interactions import (
    active_design_baseline_sweep,
    active_design_sweep,
    capacity_competition_sweep,
    cooperative_routing_metrics,
    context_stability_metrics,
    context_stability_sweep,
    developmental_bootstrap_metrics,
    factorial_interaction_phase_diagram,
    gating_metrics,
    higher_order_interaction_metrics,
    interaction_recovery_suite,
    mechanism_death_metrics,
    noisy_recovery_sweep,
    redundant_paths_metrics,
    shared_atom_reuse_metrics,
    synergistic_paths_metrics,
    table_aliasing_metrics,
)
from mnf.interactions import (
    build_interaction_matrix,
    factorial_from_callable,
    pairwise_gradient_coupling,
)
from mnf.benchmarks.interactions.gating_mechanism import gated_worker_behavior
from mnf.benchmarks.interactions.redundant_paths import redundant_or_behavior


def run(
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    coactivations: Sequence[float] = (0.01, 0.05, 0.15, 0.4),
    decoder_cosines: Sequence[float] = (0.0, 0.25, 0.5, 0.9),
    overlaps: Sequence[float] = (0.0, 0.5, 1.0),
    redundancy_weights: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    synergy_weights: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    noisy_recovery_noise_levels: Sequence[float] = (0.0, 0.01, 0.03, 0.05, 0.08, 0.12, 0.2),
    noisy_recovery_seeds: Sequence[int] = tuple(range(50)),
    active_design_noise_levels: Sequence[float] = (0.0, 0.005, 0.01, 0.02, 0.03),
    active_design_seeds: Sequence[int] = tuple(range(20)),
    active_baseline_budgets: Sequence[int] = (22, 32, 64, 128, 256),
) -> dict[str, object]:
    bootstrap_rows = [developmental_bootstrap_metrics(seed=s) for s in seeds]
    death_rows = [mechanism_death_metrics(seed=s) for s in seeds]
    capacity = capacity_competition_sweep(coactivations, decoder_cosines, overlaps)
    phase = factorial_interaction_phase_diagram(redundancy_weights, synergy_weights)
    recovery = interaction_recovery_suite(seed=seeds[0] if seeds else 0)
    noisy_recovery = noisy_recovery_sweep(noisy_recovery_noise_levels, noisy_recovery_seeds)
    active_design = active_design_sweep(active_design_noise_levels, active_design_seeds)
    active_baselines = active_design_baseline_sweep(
        noise_levels=(0.0, 0.02, 0.03),
        budgets=active_baseline_budgets,
        seeds=active_design_seeds,
    )

    grad = pairwise_gradient_coupling(
        [
            np.array([1.0, 0.3, 0.0]),
            np.array([0.8, 0.2, 0.0]),
            np.array([-0.6, 0.1, 0.0]),
        ]
    )
    matrix = build_interaction_matrix(
        ["left_path", "right_path"],
        factorials={("left_path", "right_path"): factorial_from_callable(redundant_or_behavior)},
        memberships=[{"shared_route": 0.7, "left_state": 1.0}, {"shared_route": 0.7, "right_state": 1.0}],
    )
    gate_matrix = build_interaction_matrix(
        ["gate", "worker"],
        factorials={("gate", "worker"): factorial_from_callable(gated_worker_behavior)},
        memberships=[{"policy_gate": 1.0}, {"policy_gate": 0.35, "worker_state": 1.0}],
    )
    capacity_rows = capacity["rows"]
    return {
        "redundant_paths": redundant_paths_metrics(),
        "synergistic_paths": synergistic_paths_metrics(),
        "gating": gating_metrics(),
        "higher_order": higher_order_interaction_metrics(),
        "shared_atom_reuse": shared_atom_reuse_metrics(),
        "cooperative_routing": cooperative_routing_metrics(),
        "context_stability": context_stability_metrics(),
        "context_stability_sweep": context_stability_sweep(),
        "table_aliasing": table_aliasing_metrics(),
        "interaction_recovery": recovery,
        "noisy_interaction_recovery": noisy_recovery,
        "active_design": active_design,
        "active_design_baselines": active_baselines,
        "developmental_bootstrap_rows": bootstrap_rows,
        "mechanism_death_rows": death_rows,
        "capacity_competition": capacity,
        "factorial_interaction_phase_diagram": phase,
        "max_capacity_competition": float(max(row["capacity_competition"] for row in capacity_rows)),
        "phase_diagram_rows": len(phase["rows"]),
        "mean_developmental_lag_steps": float(np.mean([row["developmental_lag_steps"] for row in bootstrap_rows])),
        "mean_suppression_delta": float(np.mean([row["suppression_delta"] for row in death_rows])),
        "gradient_coupling": grad.tolist(),
        "redundant_interaction_matrix": matrix.as_dict(),
        "gating_interaction_matrix": gate_matrix.as_dict(),
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
