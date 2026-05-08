from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mnf.experiments import run_absorption_demo, run_cyclic_demo, run_ground_truth_recovery, run_random_control_demo, run_superposition_phase


def main() -> None:
    out = {
        "ground_truth_recovery": run_ground_truth_recovery.run(),
        "cyclic_demo": run_cyclic_demo.run(),
        "absorption_demo": run_absorption_demo.run(),
        "random_control_demo": run_random_control_demo.run(),
        "superposition_phase_first_rows": run_superposition_phase.run()["rows"][:5],
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
