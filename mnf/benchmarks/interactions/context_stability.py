from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from mnf.interactions import (
    classify_pairwise_interaction,
    evaluate_design,
    pairwise_effects_from_observations,
    pairwise_factorial_design,
)


DEFAULT_CONTEXT_NAMES = ("left", "right", "switch", "additive")


def context_shift_behavior(state: Mapping[str, bool]) -> float:
    """Pair interaction whose label depends on the background switch.

    With `switch=True`, `left` and `right` are additive. With `switch=False`,
    the same pair has an extra AND term. This is a minimal counterexample to
    context-free claims from one pairwise all-on intervention design.
    """

    left = float(state["left"])
    right = float(state["right"])
    switch = float(state["switch"])
    additive = 0.25 * float(state["additive"])
    return left + right + (1.0 - switch) * left * right + additive


def context_shift_behavior_factory(
    synergy_on: float = 0.0,
    synergy_off: float = 1.0,
    additive_weight: float = 0.25,
) -> Callable[[Mapping[str, bool]], float]:
    def behavior(state: Mapping[str, bool]) -> float:
        left = float(state["left"])
        right = float(state["right"])
        switch = float(state["switch"])
        additive = additive_weight * float(state["additive"])
        synergy = switch * synergy_on + (1.0 - switch) * synergy_off
        return left + right + synergy * left * right + additive

    return behavior


def context_labels(
    names: Sequence[str] = DEFAULT_CONTEXT_NAMES,
    context_on: bool = True,
    tol: float = 1e-7,
    behavior_fn: Callable[[Mapping[str, bool]], float] = context_shift_behavior,
) -> dict[tuple[str, str], str]:
    design = pairwise_factorial_design(names, context_on=context_on)
    observations = evaluate_design(names, behavior_fn, design)
    factorials = pairwise_effects_from_observations(names, observations, context_on=context_on)
    return {pair: classify_pairwise_interaction(effects, tol=tol) for pair, effects in factorials.items()}


def context_stability_metrics(
    names: Sequence[str] = DEFAULT_CONTEXT_NAMES,
    tol: float = 1e-7,
    behavior_fn: Callable[[Mapping[str, bool]], float] = context_shift_behavior,
) -> dict[str, object]:
    on_labels = context_labels(names, context_on=True, tol=tol, behavior_fn=behavior_fn)
    off_labels = context_labels(names, context_on=False, tol=tol, behavior_fn=behavior_fn)
    changed = {
        pair: {"context_on": on_labels[pair], "context_off": off_labels[pair]}
        for pair in on_labels
        if on_labels[pair] != off_labels[pair]
    }
    focal = ("left", "right")
    n_pairs = len(on_labels)
    return {
        "n_pairs": n_pairs,
        "n_changed_pairs": len(changed),
        "stable_label_rate": (n_pairs - len(changed)) / n_pairs if n_pairs else 1.0,
        "focal_pair": list(focal),
        "focal_context_on_label": on_labels[focal],
        "focal_context_off_label": off_labels[focal],
        "changed_pairs": {
            f"{pair[0]}::{pair[1]}": labels
            for pair, labels in changed.items()
        },
    }


def context_stability_sweep(
    synergy_on_values: Sequence[float] = (0.0, 0.25, 0.5, 1.0),
    synergy_off_values: Sequence[float] = (0.0, 0.25, 0.5, 1.0),
    tol: float = 1e-7,
) -> dict[str, object]:
    rows = []
    for synergy_on in synergy_on_values:
        for synergy_off in synergy_off_values:
            behavior_fn = context_shift_behavior_factory(synergy_on=synergy_on, synergy_off=synergy_off)
            metrics = context_stability_metrics(tol=tol, behavior_fn=behavior_fn)
            rows.append(
                {
                    "synergy_on": float(synergy_on),
                    "synergy_off": float(synergy_off),
                    "stable_label_rate": metrics["stable_label_rate"],
                    "n_changed_pairs": metrics["n_changed_pairs"],
                    "focal_context_on_label": metrics["focal_context_on_label"],
                    "focal_context_off_label": metrics["focal_context_off_label"],
                }
            )
    changed_rate = sum(1 for row in rows if row["n_changed_pairs"] > 0) / len(rows) if rows else 0.0
    return {
        "rows": rows,
        "n_rows": len(rows),
        "changed_context_rate": float(changed_rate),
    }
