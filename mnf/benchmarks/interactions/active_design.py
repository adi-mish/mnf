from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from mnf.benchmarks.interactions.recovery_suite import (
    DEFAULT_RECOVERY_NAMES,
    expected_recovery_labels,
    interaction_recovery_behavior,
)
from mnf.interactions import active_interaction_discovery, label_recovery_report
from mnf.interactions.active import active_pair_statuses
from mnf.interactions.design import pairwise_factorial_design


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


def repeated_design_trial(
    noise: float = 0.0,
    seed: int = 0,
    budget: int = 64,
    mode: str = "uniform",
    names: Sequence[str] = DEFAULT_RECOVERY_NAMES,
    z: float = 2.0,
) -> dict[str, float | int | bool | str]:
    if budget < 0:
        raise ValueError("budget must be non-negative")
    if mode not in {"uniform", "random"}:
        raise ValueError("mode must be 'uniform' or 'random'")

    rng = np.random.default_rng(seed)
    candidates = pairwise_factorial_design(names)
    observations: dict[tuple[int, ...], list[float]] = {}
    for t in range(budget):
        if mode == "uniform":
            state = candidates[t % len(candidates)]
        else:
            state = candidates[int(rng.integers(0, len(candidates)))]
        value = float(interaction_recovery_behavior(state.as_dict(names)))
        if noise > 0:
            value += float(noise * rng.normal())
        observations.setdefault(state.values, []).append(value)

    statuses = active_pair_statuses(names, observations, noise, z=z)
    pred_labels = {
        pair: status.label
        for pair, status in statuses.items()
        if status.ready
    }
    report = label_recovery_report(expected_recovery_labels(names), pred_labels)
    return {
        "mode": mode,
        "noise": float(noise),
        "seed": int(seed),
        "budget": int(budget),
        "measurements": int(budget),
        "unique_states": int(len(observations)),
        "all_stable": bool(statuses and all(status.stable for status in statuses.values())),
        "n_ready_pairs": int(sum(status.ready for status in statuses.values())),
        "n_stable_pairs": int(sum(status.stable for status in statuses.values())),
        "accuracy": float(report.accuracy),
        "f1": float(report.f1),
    }


def active_design_baseline_sweep(
    noise_levels: Sequence[float] = (0.0, 0.02, 0.03),
    budgets: Sequence[int] = (22, 32, 64, 128, 256),
    seeds: Sequence[int] = tuple(range(20)),
) -> dict[str, object]:
    rows = []
    for noise in noise_levels:
        for budget in budgets:
            active_trials = [
                active_design_trial(noise, seed, max_measurements=budget)
                for seed in seeds
            ]
            uniform_trials = [
                repeated_design_trial(noise, seed, budget=budget, mode="uniform")
                for seed in seeds
            ]
            random_trials = [
                repeated_design_trial(noise, seed, budget=budget, mode="random")
                for seed in seeds
            ]
            rows.append(
                {
                    "noise": float(noise),
                    "budget": int(budget),
                    "active_mean_measurements": float(np.mean([trial["measurements"] for trial in active_trials])),
                    "active_stable_rate": float(np.mean([float(trial["all_stable"]) for trial in active_trials])),
                    "active_mean_accuracy": float(np.mean([trial["accuracy"] for trial in active_trials])),
                    "active_mean_f1": float(np.mean([trial["f1"] for trial in active_trials])),
                    "uniform_stable_rate": float(np.mean([float(trial["all_stable"]) for trial in uniform_trials])),
                    "uniform_mean_accuracy": float(np.mean([trial["accuracy"] for trial in uniform_trials])),
                    "uniform_mean_f1": float(np.mean([trial["f1"] for trial in uniform_trials])),
                    "random_stable_rate": float(np.mean([float(trial["all_stable"]) for trial in random_trials])),
                    "random_mean_accuracy": float(np.mean([trial["accuracy"] for trial in random_trials])),
                    "random_mean_f1": float(np.mean([trial["f1"] for trial in random_trials])),
                }
            )
    return {"rows": rows}
