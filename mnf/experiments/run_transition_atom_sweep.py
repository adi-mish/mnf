from __future__ import annotations

import json
from collections.abc import Sequence

import numpy as np

from mnf.benchmarks.transition_only import make_transition_only_dataset, compare_global_linear_vs_molt


def _mean(rows: list[dict[str, float]], key: str) -> float:
    return float(np.mean([row[key] for row in rows]))


def run(
    seed: int = 0,
    n: int = 3000,
    gate_separations: Sequence[float] = (0.75, 1.5, 2.5, 4.0),
    noise_levels: Sequence[float] = (0.0, 0.05, 0.2),
    seeds: Sequence[int] = (0, 1, 2),
) -> dict[str, object]:
    """Sweep a transition-only benchmark across gate separability and noise."""

    rows = []
    for gate_separation in gate_separations:
        for noise in noise_levels:
            metrics = []
            for offset in seeds:
                ds = make_transition_only_dataset(
                    n=n,
                    gate_separation=gate_separation,
                    noise=noise,
                    seed=seed + offset,
                )
                metrics.append(compare_global_linear_vs_molt(ds, seed=seed + offset))
            global_mse = _mean(metrics, "global_linear_mse")
            molt_mse = _mean(metrics, "molt_mse")
            rows.append(
                {
                    "gate_separation": float(gate_separation),
                    "noise": float(noise),
                    "global_linear_mse": global_mse,
                    "molt_mse": molt_mse,
                    "molt_to_global_mse_ratio": float(molt_mse / global_mse),
                    "molt_description_length": _mean(metrics, "molt_description_length"),
                }
            )
    return {"rows": rows}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
