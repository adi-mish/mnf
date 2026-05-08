from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mnf.experiments import (
    run_adversarial_controls,
    run_absorption_phase,
    run_baseline_comparison,
    run_cyclic_noise_sweep,
    run_ground_truth_suite,
    run_superposition_phase,
    run_transition_atom_sweep,
)


def main() -> None:
    out = {
        "absorption_phase": run_absorption_phase.run(n=5000, seeds=(0, 1, 2, 3, 4)),
        "adversarial_controls": run_adversarial_controls.run(),
        "baseline_comparison": run_baseline_comparison.run(),
        "ground_truth_suite": run_ground_truth_suite.run(),
        "cyclic_noise_sweep": run_cyclic_noise_sweep.run(n=3000, seeds=(0, 1, 2, 3, 4)),
        "transition_atom_sweep": run_transition_atom_sweep.run(n=6000, seeds=(0, 1, 2, 3, 4)),
        "superposition_phase": run_superposition_phase.run(),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
