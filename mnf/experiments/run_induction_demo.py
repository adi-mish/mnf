from __future__ import annotations

import json
from collections.abc import Sequence

import numpy as np

from mnf.benchmarks.induction import compare_induction_vs_memorization


def run(seed: int = 0, seeds: Sequence[int] = (0, 1, 2, 3, 4)) -> dict[str, object]:
    rows = []
    for offset in seeds:
        metrics = compare_induction_vs_memorization(seed=seed + offset)
        metrics["seed"] = int(seed + offset)
        rows.append(metrics)
    return {
        "rows": rows,
        "mean_memorizer_test_accuracy": float(np.mean([r["memorizer_test_accuracy"] for r in rows])),
        "mean_copy_test_accuracy": float(np.mean([r["copy_test_accuracy"] for r in rows])),
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
