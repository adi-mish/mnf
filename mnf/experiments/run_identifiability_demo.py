from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mnf.benchmarks import atom_splitting_identifiability_metrics


def run() -> dict[str, object]:
    return {"atom_splitting": atom_splitting_identifiability_metrics()}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
