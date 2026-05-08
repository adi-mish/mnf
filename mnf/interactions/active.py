from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from mnf.interactions.design import InterventionState, pairwise_factorial_design
from mnf.interactions.factorial_effects import FactorialEffects, classify_pairwise_interaction


@dataclass(frozen=True)
class ActivePairStatus:
    pair: tuple[str, str]
    ready: bool
    stable: bool
    label: str
    tolerance: float
    min_cell_count: int
    margin: float

    def as_dict(self) -> dict[str, object]:
        return {
            "pair": list(self.pair),
            "ready": self.ready,
            "stable": self.stable,
            "label": self.label,
            "tolerance": self.tolerance,
            "min_cell_count": self.min_cell_count,
            "margin": self.margin,
        }


def pair_cell_states(
    names: Sequence[str],
    pair: tuple[str, str],
    context_on: bool = True,
) -> dict[str, tuple[int, ...]]:
    n = len(names)
    index = {name: i for i, name in enumerate(names)}
    i, j = index[pair[0]], index[pair[1]]
    context_value = 1 if context_on else 0

    def state_for(a: int, b: int) -> tuple[int, ...]:
        values = [context_value] * n
        values[i] = a
        values[j] = b
        return tuple(values)

    return {
        "y00": state_for(0, 0),
        "y10": state_for(1, 0),
        "y01": state_for(0, 1),
        "y11": state_for(1, 1),
    }


def cell_means(observations: Mapping[tuple[int, ...], Sequence[float]]) -> dict[tuple[int, ...], float]:
    return {state: float(np.mean(values)) for state, values in observations.items() if len(values) > 0}


def _cell_radius(noise: float, count: int, z: float = 2.0) -> float:
    if count <= 0:
        return float("inf")
    return float(z * noise / np.sqrt(count))


def _pair_margin(effects: FactorialEffects) -> float:
    interaction_margin = max(abs(effects.synergy), effects.redundancy_score, effects.competition_score)
    return float(max(abs(effects.joint_effect), interaction_margin))


def pair_status(
    names: Sequence[str],
    pair: tuple[str, str],
    observations: Mapping[tuple[int, ...], Sequence[float]],
    noise: float,
    z: float = 2.0,
    context_on: bool = True,
) -> ActivePairStatus:
    states = pair_cell_states(names, pair, context_on=context_on)
    counts = {cell: len(observations.get(state, ())) for cell, state in states.items()}
    min_count = min(counts.values())
    if min_count == 0:
        return ActivePairStatus(
            pair=pair,
            ready=False,
            stable=False,
            label="unknown",
            tolerance=float("inf"),
            min_cell_count=0,
            margin=0.0,
        )
    means = cell_means(observations)
    effects = FactorialEffects(
        y00=means[states["y00"]],
        y10=means[states["y10"]],
        y01=means[states["y01"]],
        y11=means[states["y11"]],
    )
    max_radius = max(_cell_radius(noise, count, z=z) for count in counts.values())
    tolerance = max(1e-9, 4.0 * max_radius)
    label = classify_pairwise_interaction(effects, tol=tolerance)
    interaction_margin = max(abs(effects.synergy), effects.redundancy_score, effects.competition_score)
    if label == "additive":
        stable = abs(effects.joint_effect) > 2.0 * tolerance and interaction_margin <= tolerance
        margin = abs(effects.joint_effect)
    elif label == "inactive":
        stable = abs(effects.joint_effect) <= tolerance and interaction_margin <= tolerance
        margin = max(abs(effects.joint_effect), interaction_margin)
    elif label == "redundant":
        stable = effects.redundancy_score > 2.0 * tolerance
        margin = effects.redundancy_score
    elif label == "competitive":
        stable = effects.competition_score > 2.0 * tolerance
        margin = effects.competition_score
    else:
        stable = effects.synergy > 2.0 * tolerance
        margin = effects.synergy
    return ActivePairStatus(
        pair=pair,
        ready=True,
        stable=bool(stable),
        label=label,
        tolerance=float(tolerance),
        min_cell_count=int(min_count),
        margin=float(margin),
    )


def active_pair_statuses(
    names: Sequence[str],
    observations: Mapping[tuple[int, ...], Sequence[float]],
    noise: float,
    z: float = 2.0,
    context_on: bool = True,
) -> dict[tuple[str, str], ActivePairStatus]:
    return {
        tuple(pair): pair_status(names, tuple(pair), observations, noise, z=z, context_on=context_on)
        for pair in combinations(names, 2)
    }


def choose_next_intervention_state(
    names: Sequence[str],
    observations: Mapping[tuple[int, ...], Sequence[float]],
    noise: float,
    z: float = 2.0,
    context_on: bool = True,
) -> InterventionState:
    candidates = pairwise_factorial_design(names, context_on=context_on)
    statuses = active_pair_statuses(names, observations, noise, z=z, context_on=context_on)
    pair_states = {pair: set(pair_cell_states(names, pair, context_on=context_on).values()) for pair in statuses}
    unobserved = [candidate for candidate in candidates if len(observations.get(candidate.values, ())) == 0]
    if unobserved:
        return max(
            unobserved,
            key=lambda candidate: sum(
                1
                for pair, status in statuses.items()
                if not status.ready and candidate.values in pair_states[pair]
            ),
        )
    best_state = candidates[0]
    best_score = -1.0
    for candidate in candidates:
        count = len(observations.get(candidate.values, ()))
        score = 0.0
        for pair, status in statuses.items():
            if status.stable or candidate.values not in pair_states[pair]:
                continue
            if not status.ready:
                missing = sum(1 for state in pair_states[pair] if len(observations.get(state, ())) == 0)
                if count == 0:
                    score += 100.0 + missing
                else:
                    score += 0.01 * missing / (1.0 + count)
            else:
                gap = max(status.margin - status.tolerance, 1e-6)
                score += 1.0 / ((1.0 + count) * gap)
        score += 1e-6 / (1.0 + count)
        if score > best_score:
            best_score = score
            best_state = candidate
    return best_state


def active_interaction_discovery(
    names: Sequence[str],
    behavior_fn: Callable[[Mapping[str, bool]], float],
    noise: float = 0.0,
    seed: int = 0,
    max_measurements: int = 256,
    z: float = 2.0,
    context_on: bool = True,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    observations: dict[tuple[int, ...], list[float]] = {}
    measurements = 0
    while measurements < max_measurements:
        statuses = active_pair_statuses(names, observations, noise, z=z, context_on=context_on)
        if statuses and all(status.stable for status in statuses.values()):
            break
        state = choose_next_intervention_state(names, observations, noise, z=z, context_on=context_on)
        value = float(behavior_fn(state.as_dict(names)))
        if noise > 0:
            value += float(noise * rng.normal())
        observations.setdefault(state.values, []).append(value)
        measurements += 1

    statuses = active_pair_statuses(names, observations, noise, z=z, context_on=context_on)
    labels = {pair: status.label for pair, status in statuses.items() if status.ready}
    return {
        "measurements": measurements,
        "unique_states": len(observations),
        "all_stable": bool(statuses and all(status.stable for status in statuses.values())),
        "n_ready_pairs": int(sum(status.ready for status in statuses.values())),
        "n_stable_pairs": int(sum(status.stable for status in statuses.values())),
        "labels": {f"{pair[0]}::{pair[1]}": label for pair, label in labels.items()},
        "statuses": {f"{pair[0]}::{pair[1]}": status.as_dict() for pair, status in statuses.items()},
    }
