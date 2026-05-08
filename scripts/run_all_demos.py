from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mnf.experiments import (
    run_absorption_demo,
    run_absorption_phase,
    run_cyclic_demo,
    run_cyclic_noise_sweep,
    run_ground_truth_recovery,
    run_ground_truth_suite,
    run_mechanismlab_demo,
    run_random_control_demo,
    run_superposition_phase,
    run_transition_atom_sweep,
)


def main() -> None:
    out = {
        "ground_truth_recovery": run_ground_truth_recovery.run(),
        "ground_truth_suite": run_ground_truth_suite.run(),
        "cyclic_demo": run_cyclic_demo.run(),
        "absorption_demo": run_absorption_demo.run(),
        "absorption_phase_first_rows": run_absorption_phase.run()["rows"][:5],
        "cyclic_noise_first_rows": run_cyclic_noise_sweep.run()["rows"][:5],
        "random_control_demo": run_random_control_demo.run(),
        "mechanismlab_demo": run_mechanismlab_demo.run(),
        "transition_atom_sweep_first_rows": run_transition_atom_sweep.run()["rows"][:5],
        "superposition_phase_first_rows": run_superposition_phase.run()["rows"][:5],
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
