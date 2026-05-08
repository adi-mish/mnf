from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations, product

from mnf.interactions.factorial_effects import FactorialEffects


@dataclass(frozen=True)
class InterventionState:
    values: tuple[int, ...]

    def __post_init__(self) -> None:
        if any(value not in (0, 1) for value in self.values):
            raise ValueError("intervention states must be binary")

    def as_dict(self, names: Sequence[str]) -> dict[str, bool]:
        if len(names) != len(self.values):
            raise ValueError("names length must match state length")
        return {name: bool(value) for name, value in zip(names, self.values)}


def full_factorial_design(n_mechanisms: int) -> tuple[InterventionState, ...]:
    return tuple(InterventionState(tuple(values)) for values in product((0, 1), repeat=n_mechanisms))


def pairwise_factorial_design(names: Sequence[str], context_on: bool = True) -> tuple[InterventionState, ...]:
    """Four-cell pairwise designs with non-pair mechanisms held fixed."""

    n = len(names)
    context_value = 1 if context_on else 0
    states: set[InterventionState] = set()
    for i, j in combinations(range(n), 2):
        for a, b in product((0, 1), repeat=2):
            values = [context_value] * n
            values[i] = a
            values[j] = b
            states.add(InterventionState(tuple(values)))
    return tuple(sorted(states, key=lambda state: state.values))


def sparse_higher_order_design(names: Sequence[str], max_order: int = 2, context_on: bool = True) -> tuple[InterventionState, ...]:
    """Baseline plus all ablations up to `max_order` mechanisms."""

    n = len(names)
    context_value = 1 if context_on else 0
    ablated_value = 1 - context_value
    states: set[InterventionState] = {InterventionState(tuple([context_value] * n))}
    for order in range(1, min(max_order, n) + 1):
        for subset in combinations(range(n), order):
            values = [context_value] * n
            for idx in subset:
                values[idx] = ablated_value
            states.add(InterventionState(tuple(values)))
    return tuple(sorted(states, key=lambda state: state.values))


def evaluate_design(
    names: Sequence[str],
    behavior_fn: Callable[[Mapping[str, bool]], float],
    design: Sequence[InterventionState],
) -> dict[tuple[int, ...], float]:
    return {state.values: float(behavior_fn(state.as_dict(names))) for state in design}


def pairwise_effects_from_observations(
    names: Sequence[str],
    observations: Mapping[tuple[int, ...], float],
    context_on: bool = True,
) -> dict[tuple[str, str], FactorialEffects]:
    n = len(names)
    context_value = 1 if context_on else 0
    out: dict[tuple[str, str], FactorialEffects] = {}
    for i, j in combinations(range(n), 2):
        def state_for(a: int, b: int) -> tuple[int, ...]:
            values = [context_value] * n
            values[i] = a
            values[j] = b
            return tuple(values)

        try:
            out[(names[i], names[j])] = FactorialEffects(
                y00=float(observations[state_for(0, 0)]),
                y10=float(observations[state_for(1, 0)]),
                y01=float(observations[state_for(0, 1)]),
                y11=float(observations[state_for(1, 1)]),
            )
        except KeyError as exc:
            raise ValueError(f"missing observation for pair {(names[i], names[j])}") from exc
    return out


def higher_order_effect_from_observations(
    names: Sequence[str],
    observations: Mapping[tuple[int, ...], float],
    subset: Sequence[str],
    context_on: bool = True,
) -> float:
    """Inclusion-exclusion contrast for a mechanism subset.

    For a pair this equals the standard synergy term. For a triple it estimates
    the part of the behavior not explained by lower-order subset terms in the
    selected intervention context.
    """

    n = len(names)
    index = {name: i for i, name in enumerate(names)}
    subset_indices = tuple(index[name] for name in subset)
    if not subset_indices:
        raise ValueError("subset must not be empty")
    context_value = 1 if context_on else 0
    total = 0.0
    k = len(subset_indices)
    for bits in product((0, 1), repeat=k):
        values = [context_value] * n
        for idx, bit in zip(subset_indices, bits):
            values[idx] = bit
        sign = (-1.0) ** (k - sum(bits))
        try:
            total += sign * float(observations[tuple(values)])
        except KeyError as exc:
            raise ValueError(f"missing observation for subset {tuple(subset)}") from exc
    return float(total)


def choose_next_pair_by_uncertainty(
    names: Sequence[str],
    uncertainty: Mapping[tuple[str, str], float],
    observed_pairs: set[tuple[str, str]] | None = None,
) -> tuple[str, str]:
    """Choose the unobserved or highest-uncertainty pair."""

    observed_pairs = observed_pairs or set()
    pairs = [tuple(pair) for pair in combinations(names, 2)]
    for pair in pairs:
        if pair not in observed_pairs and (pair[1], pair[0]) not in observed_pairs:
            return pair
    if not pairs:
        raise ValueError("at least two mechanisms are required")
    return max(pairs, key=lambda pair: float(uncertainty.get(pair, uncertainty.get((pair[1], pair[0]), 0.0))))
