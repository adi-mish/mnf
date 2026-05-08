from __future__ import annotations

from collections.abc import Mapping, Sequence

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


def context_labels(
    names: Sequence[str] = DEFAULT_CONTEXT_NAMES,
    context_on: bool = True,
    tol: float = 1e-7,
) -> dict[tuple[str, str], str]:
    design = pairwise_factorial_design(names, context_on=context_on)
    observations = evaluate_design(names, context_shift_behavior, design)
    factorials = pairwise_effects_from_observations(names, observations, context_on=context_on)
    return {pair: classify_pairwise_interaction(effects, tol=tol) for pair, effects in factorials.items()}


def context_stability_metrics(
    names: Sequence[str] = DEFAULT_CONTEXT_NAMES,
    tol: float = 1e-7,
) -> dict[str, object]:
    on_labels = context_labels(names, context_on=True, tol=tol)
    off_labels = context_labels(names, context_on=False, tol=tol)
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
