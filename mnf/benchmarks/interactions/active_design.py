from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from mnf.benchmarks.interactions.recovery_suite import (
    DEFAULT_RECOVERY_NAMES,
    expected_recovery_labels,
    interaction_recovery_behavior,
)
from mnf.interactions import active_interaction_discovery, label_recovery_report


def active_design_trial(
    noise: float = 0.0,
    seed: int = 0,
    names: Sequence[str] = DEFAULT_RECOVERY_NAMES,
    max_measurements: int = 256,
    z: float = 2.0,
) -> dict[str, float | int | bool]:
    discovery = active_interaction_discovery(
        names,
        interaction_recovery_behavior,
        noise=noise,
        seed=seed,
        max_measurements=max_measurements,
        z=z,
    )
    pred_labels = {
        tuple(key.split("::")): value
        for key, value in discovery["labels"].items()
    }
    report = label_recovery_report(expected_recovery_labels(names), pred_labels)
    return {
        "noise": float(noise),
        "seed": int(seed),
        "measurements": int(discovery["measurements"]),
        "unique_states": int(discovery["unique_states"]),
        "all_stable": bool(discovery["all_stable"]),
        "n_ready_pairs": int(discovery["n_ready_pairs"]),
        "n_stable_pairs": int(discovery["n_stable_pairs"]),
        "accuracy": float(report.accuracy),
        "f1": float(report.f1),
    }


def active_design_sweep(
    noise_levels: Sequence[float] = (0.0, 0.005, 0.01, 0.02, 0.03),
    seeds: Sequence[int] = tuple(range(50)),
    max_measurements: int = 256,
) -> dict[str, object]:
    rows = []
    for noise in noise_levels:
        trials = [active_design_trial(noise, seed, max_measurements=max_measurements) for seed in seeds]
        rows.append(
            {
                "noise": float(noise),
                "mean_measurements": float(np.mean([trial["measurements"] for trial in trials])),
                "mean_unique_states": float(np.mean([trial["unique_states"] for trial in trials])),
                "all_stable_rate": float(np.mean([float(trial["all_stable"]) for trial in trials])),
                "mean_accuracy": float(np.mean([trial["accuracy"] for trial in trials])),
                "mean_f1": float(np.mean([trial["f1"] for trial in trials])),
            }
        )
    return {"rows": rows}
