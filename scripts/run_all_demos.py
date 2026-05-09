from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mnf.experiments import (
    run_adversarial_controls,
    run_absorption_demo,
    run_absorption_phase,
    run_atlas_suite,
    run_baseline_comparison,
    run_certificate_demo,
    run_cyclic_baseline_comparison,
    run_cyclic_demo,
    run_cyclic_noise_sweep,
    run_feature_baseline_suite,
    run_ground_truth_recovery,
    run_ground_truth_suite,
    run_identifiability_demo,
    run_induction_demo,
    run_interaction_suite,
    run_mechanismlab_demo,
    run_random_control_demo,
    run_superposition_phase,
    run_tiny_transformer_demo,
    run_training_emergence,
    run_transition_atom_sweep,
)


def main() -> None:
    out = {
        "ground_truth_recovery": run_ground_truth_recovery.run(),
        "ground_truth_suite": run_ground_truth_suite.run(),
        "identifiability_demo": run_identifiability_demo.run(),
        "induction_demo": run_induction_demo.run(),
        "interaction_suite": run_interaction_suite.run(),
        "atlas_suite": run_atlas_suite.run(),
        "certificate_demo": run_certificate_demo.run(),
        "adversarial_controls": run_adversarial_controls.run(),
        "baseline_comparison": run_baseline_comparison.run(),
        "feature_baseline_suite": run_feature_baseline_suite.run(seeds=(0, 1)),
        "cyclic_baseline_comparison": run_cyclic_baseline_comparison.run(),
        "cyclic_demo": run_cyclic_demo.run(),
        "absorption_demo": run_absorption_demo.run(),
        "absorption_phase_first_rows": run_absorption_phase.run()["rows"][:5],
        "cyclic_noise_first_rows": run_cyclic_noise_sweep.run()["rows"][:5],
        "random_control_demo": run_random_control_demo.run(),
        "mechanismlab_demo": run_mechanismlab_demo.run(),
        "transition_atom_sweep_first_rows": run_transition_atom_sweep.run()["rows"][:5],
        "tiny_transformer_demo": run_tiny_transformer_demo.run(seeds=(0,), steps=120),
        "training_emergence": run_training_emergence.run(),
        "superposition_phase_first_rows": run_superposition_phase.run()["rows"][:5],
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
