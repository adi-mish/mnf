from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from mnf.interactions import FactorialEffects, InteractionMatrix, build_interaction_matrix, factorial_from_callable


def infer_pairwise_factorials(
    behavior_by_pair: Mapping[tuple[str, str], Callable[[bool, bool], float]],
) -> dict[tuple[str, str], FactorialEffects]:
    """Run pairwise on/off intervention probes for named mechanism pairs."""

    return {pair: factorial_from_callable(fn) for pair, fn in behavior_by_pair.items()}


def mechanism_ecology_discovery(
    names: Sequence[str],
    behavior_by_pair: Mapping[tuple[str, str], Callable[[bool, bool], float]],
    memberships: Sequence[Mapping[str, float]] | None = None,
) -> InteractionMatrix:
    """Minimal MED entry point for synthetic executable mechanisms.

    This is not a full real-model discovery algorithm. It is the executable
    core used by MechanismLab 2.0 benchmarks: propose candidate mechanisms,
    run pairwise factorial interventions, and return the interaction matrix
    that later stages can prune with shared MDL.
    """

    factorials = infer_pairwise_factorials(behavior_by_pair)
    return build_interaction_matrix(names=names, factorials=factorials, memberships=memberships)
