from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SharedMDLDecision:
    kept_atoms: tuple[str, ...]
    dropped_atoms: tuple[str, ...]
    shared_cost: float
    independent_cost: float
    gain: float

    def as_dict(self) -> dict[str, object]:
        return {
            "kept_atoms": list(self.kept_atoms),
            "dropped_atoms": list(self.dropped_atoms),
            "shared_cost": self.shared_cost,
            "independent_cost": self.independent_cost,
            "gain": self.gain,
        }


def greedy_shared_mdl_prune(
    memberships: Sequence[Mapping[str, float]],
    atom_costs: Mapping[str, float] | None = None,
    mechanism_cost: float = 1.0,
    threshold: float = 1e-12,
) -> SharedMDLDecision:
    """Keep atoms used by at least one mechanism and report shared-MDL gain."""

    atom_costs = atom_costs or {}
    active_by_mechanism: list[set[str]] = []
    used: set[str] = set()
    for membership in memberships:
        active = {name for name, weight in membership.items() if abs(float(weight)) > threshold}
        active_by_mechanism.append(active)
        used |= active

    def cost(name: str) -> float:
        return float(atom_costs.get(name, 1.0))

    shared_cost = float(len(memberships) * mechanism_cost + sum(cost(name) for name in used))
    independent_cost = float(
        len(memberships) * mechanism_cost
        + sum(cost(name) for active in active_by_mechanism for name in active)
    )
    all_atoms = set(atom_costs) | used
    dropped = tuple(sorted(all_atoms - used))
    return SharedMDLDecision(
        kept_atoms=tuple(sorted(used)),
        dropped_atoms=dropped,
        shared_cost=shared_cost,
        independent_cost=independent_cost,
        gain=float(independent_cost - shared_cost),
    )


def membership_overlap(memberships: Sequence[Mapping[str, float]]) -> float:
    if len(memberships) < 2:
        return 0.0
    atoms = sorted({atom for membership in memberships for atom in membership})
    if not atoms:
        return 0.0
    rows = np.array([[float(membership.get(atom, 0.0)) for atom in atoms] for membership in memberships])
    norms = np.linalg.norm(rows, axis=1)
    overlaps = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            denom = norms[i] * norms[j]
            overlaps.append(0.0 if denom <= 1e-12 else float(np.dot(rows[i], rows[j]) / denom))
    return float(np.mean(overlaps)) if overlaps else 0.0
