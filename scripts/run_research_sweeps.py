from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mnf.experiments import (
    run_adversarial_controls,
    run_absorption_phase,
    run_baseline_comparison,
    run_cyclic_baseline_comparison,
    run_cyclic_noise_sweep,
    run_ground_truth_suite,
    run_induction_demo,
    run_interaction_suite,
    run_superposition_phase,
    run_training_emergence,
    run_transition_atom_sweep,
)


def main() -> None:
    out = {
        "absorption_phase": run_absorption_phase.run(n=5000, seeds=(0, 1, 2, 3, 4)),
        "adversarial_controls": run_adversarial_controls.run(),
        "baseline_comparison": run_baseline_comparison.run(),
        "cyclic_baseline_comparison": run_cyclic_baseline_comparison.run(),
        "ground_truth_suite": run_ground_truth_suite.run(),
        "induction_demo": run_induction_demo.run(seeds=tuple(range(20))),
        "interaction_suite": run_interaction_suite.run(
            seeds=tuple(range(20)),
            coactivations=(0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.4, 0.7),
            decoder_cosines=(0.0, 0.15, 0.3, 0.5, 0.7, 0.9, 1.0),
            overlaps=(0.0, 0.25, 0.5, 0.75, 1.0),
            redundancy_weights=(0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5),
            synergy_weights=(0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5),
            noisy_recovery_noise_levels=(0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3),
            noisy_recovery_seeds=tuple(range(200)),
            active_design_noise_levels=(0.0, 0.005, 0.01, 0.02, 0.03, 0.04),
            active_design_seeds=tuple(range(100)),
        ),
        "cyclic_noise_sweep": run_cyclic_noise_sweep.run(n=3000, seeds=(0, 1, 2, 3, 4)),
        "transition_atom_sweep": run_transition_atom_sweep.run(n=6000, seeds=(0, 1, 2, 3, 4)),
        "training_emergence": run_training_emergence.run(seeds=(0, 1, 2, 3, 4)),
        "superposition_phase": run_superposition_phase.run(),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
