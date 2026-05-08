from __future__ import annotations

import json
from mnf.benchmarks.random_control import random_vs_trained_control


def run(seed: int = 0) -> dict[str, float]:
    return random_vs_trained_control(seed=seed)


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
