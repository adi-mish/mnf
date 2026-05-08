from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from mnf.benchmarks.interactions.recovery_suite import (
    expected_recovery_labels,
    interaction_recovery_behavior,
)
from mnf.interactions import (
    classify_pairwise_interaction,
    evaluate_design,
    label_recovery_report,
    pairwise_contrast_error_bounds,
    pairwise_effects_from_observations,
    pairwise_factorial_design,
)


DEFAULT_RECOVERY_NAMES = (
    "redundant_left",
    "redundant_right",
    "gate",
    "worker",
    "competitor",
    "additive",
)


def noisy_recovery_trial(
    noise: float,
    seed: int = 0,
    names: Sequence[str] = DEFAULT_RECOVERY_NAMES,
) -> dict[str, float]:
    design = pairwise_factorial_design(names)
    clean = evaluate_design(names, interaction_recovery_behavior, design)
    rng = np.random.default_rng(seed)
    noisy = {state: value + noise * rng.normal() for state, value in clean.items()}
    max_abs_cell_error = max(abs(noisy[state] - clean[state]) for state in clean) if clean else 0.0
    factorials = pairwise_effects_from_observations(names, noisy)
    bounds = pairwise_contrast_error_bounds(max_abs_cell_error)
    contrast_tol = max(1e-7, bounds["synergy"])
    pred_labels = {pair: classify_pairwise_interaction(effects, tol=contrast_tol) for pair, effects in factorials.items()}
    report = label_recovery_report(expected_recovery_labels(names), pred_labels)
    return {
        "noise": float(noise),
        "seed": int(seed),
        "accuracy": float(report.accuracy),
        "f1": float(report.f1),
        "all_correct": float(report.accuracy == 1.0),
        "max_abs_cell_error": float(max_abs_cell_error),
        "synergy_error_bound": float(bounds["synergy"]),
        "gate_error_bound": float(bounds["gate_m_to_n"]),
        "classification_tolerance": float(contrast_tol),
    }


def noisy_recovery_sweep(
    noise_levels: Sequence[float] = (0.0, 0.01, 0.03, 0.05, 0.08, 0.12, 0.2),
    seeds: Sequence[int] = tuple(range(50)),
) -> dict[str, object]:
    rows = []
    for noise in noise_levels:
        trials = [noisy_recovery_trial(noise, seed) for seed in seeds]
        rows.append(
            {
                "noise": float(noise),
                "mean_accuracy": float(np.mean([trial["accuracy"] for trial in trials])),
                "mean_f1": float(np.mean([trial["f1"] for trial in trials])),
                "all_correct_rate": float(np.mean([trial["all_correct"] for trial in trials])),
                "mean_max_abs_cell_error": float(np.mean([trial["max_abs_cell_error"] for trial in trials])),
                "mean_synergy_error_bound": float(np.mean([trial["synergy_error_bound"] for trial in trials])),
                "mean_classification_tolerance": float(np.mean([trial["classification_tolerance"] for trial in trials])),
            }
        )
    return {"rows": rows}
