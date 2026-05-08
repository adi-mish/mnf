from __future__ import annotations

import json
from collections.abc import Sequence

import numpy as np

from mnf.benchmarks.cyclic import make_weekday_rotation_dataset, weekday_space
from mnf.charts.cyclic import CyclicChart
from mnf.baselines.cyclic import compare_cyclic_baselines


def run(
    seed: int = 0,
    n: int = 1200,
    noise_levels: Sequence[float] = (0.0, 0.03, 0.08, 0.15, 0.3),
    seeds: Sequence[int] = (0, 1, 2),
) -> dict[str, object]:
    """Sweep cyclic chart recovery under increasing activation noise."""

    rows = []
    for noise in noise_levels:
        base_errors = []
        rotation_errors = []
        baseline = compare_cyclic_baselines(n=n, noise=noise, seed=seed)
        for offset in seeds:
            ds = make_weekday_rotation_dataset(n=n, noise=noise, seed=seed + offset)
            chart = CyclicChart.fit(ds.activations, ds.labels, weekday_space)
            pred = chart.encode_label(ds.activations)
            base_errors.append(sum(a != b for a, b in zip(pred, ds.labels)) / len(ds.labels))
            rotation_errors.append(chart.intervention_error_after_rotation(ds.activations, ds.labels, ds.rotation_steps))
        rows.append(
            {
                "noise": float(noise),
                "base_label_error": float(np.mean(base_errors)),
                "rotation_intervention_error": float(np.mean(rotation_errors)),
                "scalar_label_error": baseline["scalar_label_error"],
                "scalar_rotation_error": baseline["scalar_rotation_error"],
            }
        )
    return {"rows": rows}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
