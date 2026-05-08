from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from mnf.interactions.capacity import capacity_competition_from_matrices
from mnf.interactions.factorial_effects import FactorialEffects
from mnf.interactions.overlap import pairwise_overlap_matrix


@dataclass(frozen=True)
class InteractionMatrix:
    names: tuple[str, ...]
    overlap: np.ndarray
    behavioral_synergy: np.ndarray
    gate: np.ndarray
    redundancy: np.ndarray
    capacity_competition: np.ndarray
    gradient_coupling: np.ndarray
    support: np.ndarray

    def as_dict(self) -> dict[str, object]:
        return {
            "names": list(self.names),
            "overlap": self.overlap.tolist(),
            "behavioral_synergy": self.behavioral_synergy.tolist(),
            "gate": self.gate.tolist(),
            "redundancy": self.redundancy.tolist(),
            "capacity_competition": self.capacity_competition.tolist(),
            "gradient_coupling": self.gradient_coupling.tolist(),
            "support": self.support.tolist(),
        }


def build_interaction_matrix(
    names: Sequence[str],
    factorials: Mapping[tuple[str, str], FactorialEffects] | None = None,
    memberships: Sequence[Mapping[str, float] | Sequence[float]] | None = None,
    capacity_memberships: np.ndarray | None = None,
    coactivation: np.ndarray | None = None,
    gram_squared: np.ndarray | None = None,
    gradient_coupling: np.ndarray | None = None,
    support: np.ndarray | None = None,
) -> InteractionMatrix:
    names_tuple = tuple(names)
    n = len(names_tuple)
    overlap = np.eye(n, dtype=float)
    if memberships is not None:
        overlap = pairwise_overlap_matrix(memberships)

    synergy = np.zeros((n, n), dtype=float)
    gate = np.zeros((n, n), dtype=float)
    redundancy = np.zeros((n, n), dtype=float)
    if factorials is not None:
        index = {name: i for i, name in enumerate(names_tuple)}
        for (m, other), effects in factorials.items():
            i = index[m]
            j = index[other]
            synergy[i, j] = effects.synergy
            synergy[j, i] = effects.synergy
            gate[i, j] = effects.gate_m_to_n
            gate[j, i] = effects.gate_n_to_m
            redundancy[i, j] = effects.redundancy_score
            redundancy[j, i] = effects.redundancy_score

    capacity = np.zeros((n, n), dtype=float)
    if capacity_memberships is not None and coactivation is not None and gram_squared is not None:
        caps = np.asarray(capacity_memberships, dtype=float)
        if caps.shape[0] != n:
            raise ValueError("capacity_memberships must have one row per mechanism")
        for i in range(n):
            for j in range(n):
                capacity[i, j] = capacity_competition_from_matrices(caps[i], caps[j], coactivation, gram_squared)

    grad = np.eye(n, dtype=float) if gradient_coupling is None else np.asarray(gradient_coupling, dtype=float)
    support_matrix = np.zeros((n, n), dtype=float) if support is None else np.asarray(support, dtype=float)
    if grad.shape != (n, n):
        raise ValueError("gradient_coupling must be K x K")
    if support_matrix.shape != (n, n):
        raise ValueError("support must be K x K")

    return InteractionMatrix(
        names=names_tuple,
        overlap=overlap,
        behavioral_synergy=synergy,
        gate=gate,
        redundancy=redundancy,
        capacity_competition=capacity,
        gradient_coupling=grad,
        support=support_matrix,
    )
