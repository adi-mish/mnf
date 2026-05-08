from __future__ import annotations

import json
from collections.abc import Sequence

import numpy as np

from mnf.benchmarks.training_dynamics import make_training_emergence_trace, emergence_lead_metrics


def run(
    seed: int = 0,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
) -> dict[str, object]:
    rows = []
    for offset in seeds:
        trace = make_training_emergence_trace(seed=seed + offset)
        metrics = emergence_lead_metrics(trace)
        metrics["seed"] = int(seed + offset)
        rows.append(metrics)
    return {
        "rows": rows,
        "mean_mechanism_lead_steps": float(np.mean([row["mechanism_lead_steps"] for row in rows])),
        "mean_correlation": float(np.mean([row["mechanism_behavior_correlation"] for row in rows])),
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
