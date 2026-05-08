from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations

import numpy as np

from mnf.discovery import mechanism_ecology_discovery_from_joint_behavior
from mnf.interactions import (
    FactorialEffects,
    bootstrap_factorial_effects,
    classify_pairwise_interaction,
    label_recovery_report,
    noisy_factorial_samples,
    pairwise_effects_from_observations,
    pairwise_factorial_design,
    evaluate_design,
)


def interaction_recovery_behavior(state: Mapping[str, bool]) -> float:
    """Joint behavior with known redundant, gated, and competitive motifs."""

    redundant = float(state["redundant_left"] or state["redundant_right"])
    gated = float(state["gate"] and state["worker"])
    competitor_penalty = 0.6 * float(state["competitor"] and state["worker"])
    additive = 0.35 * float(state["additive"])
    return redundant + gated + additive - competitor_penalty


def _expected_label(pair: tuple[str, str]) -> str:
    names = set(pair)
    if names == {"redundant_left", "redundant_right"}:
        return "redundant"
    if names == {"gate", "worker"}:
        return "synergistic_or_gated"
    if names == {"competitor", "worker"}:
        return "competitive"
    return "additive"


def _predicted_labels(factorials: Mapping[tuple[str, str], FactorialEffects]) -> dict[tuple[str, str], str]:
    return {pair: classify_pairwise_interaction(effects, tol=1e-7) for pair, effects in factorials.items()}


def interaction_recovery_suite(
    names: Sequence[str] = (
        "redundant_left",
        "redundant_right",
        "gate",
        "worker",
        "competitor",
        "additive",
    ),
    bootstrap_n: int = 400,
    seed: int = 0,
) -> dict[str, object]:
    design = pairwise_factorial_design(names)
    observations = evaluate_design(names, interaction_recovery_behavior, design)
    factorials = pairwise_effects_from_observations(names, observations)
    pred_labels = _predicted_labels(factorials)
    true_labels = {pair: _expected_label(pair) for pair in combinations(names, 2)}
    report = label_recovery_report(true_labels, pred_labels)
    matrix = mechanism_ecology_discovery_from_joint_behavior(
        names,
        interaction_recovery_behavior,
        memberships=[
            {"route": 0.8, "redundant_left_state": 1.0},
            {"route": 0.8, "redundant_right_state": 1.0},
            {"gate_state": 1.0},
            {"gate_state": 0.35, "worker_state": 1.0},
            {"worker_state": 0.4, "competitor_state": 1.0},
            {"additive_state": 1.0},
        ],
    )

    gate_samples = noisy_factorial_samples(factorials[("gate", "worker")], n=256, noise=0.04, seed=seed)
    gate_ci = bootstrap_factorial_effects(gate_samples, n_boot=bootstrap_n, seed=seed)
    redundancy_samples = noisy_factorial_samples(
        factorials[("redundant_left", "redundant_right")],
        n=256,
        noise=0.04,
        seed=seed + 1,
    )
    redundancy_ci = bootstrap_factorial_effects(redundancy_samples, n_boot=bootstrap_n, seed=seed + 1)

    label_counts: dict[str, int] = {}
    for label in pred_labels.values():
        label_counts[label] = label_counts.get(label, 0) + 1
    nonzero_synergy = int(np.sum(np.abs(matrix.behavioral_synergy) > 1e-7))
    return {
        "design_size": len(design),
        "n_pairs": len(pred_labels),
        "recovery": report.as_dict(),
        "predicted_label_counts": label_counts,
        "gate_synergy_ci": {key: interval.as_dict() for key, interval in gate_ci.items()},
        "redundancy_ci": {key: interval.as_dict() for key, interval in redundancy_ci.items()},
        "nonzero_synergy_matrix_entries": nonzero_synergy,
        "interaction_matrix": matrix.as_dict(),
    }
