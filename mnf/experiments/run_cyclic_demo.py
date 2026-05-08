from __future__ import annotations

import json
from mnf.benchmarks.cyclic import make_weekday_rotation_dataset, weekday_space
from mnf.charts.cyclic import CyclicChart


def run(seed: int = 0) -> dict[str, float]:
    ds = make_weekday_rotation_dataset(seed=seed)
    chart = CyclicChart.fit(ds.activations, ds.labels, weekday_space)
    base_pred = chart.encode_label(ds.activations)
    base_err = sum(a != b for a, b in zip(base_pred, ds.labels)) / len(ds.labels)
    rot_err = chart.intervention_error_after_rotation(ds.activations, ds.labels, ds.rotation_steps)
    return {"base_label_error": base_err, "rotation_intervention_error": rot_err}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
