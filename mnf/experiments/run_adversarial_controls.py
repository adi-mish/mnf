from __future__ import annotations

import json

from mnf.benchmarks.memorization import memorization_control
from mnf.benchmarks.random_control import random_vs_trained_control
from mnf.benchmarks.shortcut import shortcut_control


def run(seed: int = 0) -> dict[str, object]:
    return {
        "random_vs_trained": random_vs_trained_control(seed=seed),
        "shortcut_spurious_correlation": shortcut_control(seed=seed),
        "memorizing_alignment": memorization_control(seed=seed),
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
