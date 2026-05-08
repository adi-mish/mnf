from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from mnf.mechanisms.fuzzy import Atom, FuzzyMechanism
from mnf.mechanisms.shared_mdl import independent_description_length, shared_description_length, shared_mdl_gain, used_atoms


@dataclass(frozen=True)
class MechanismEcosystem:
    atoms: Mapping[str, Atom]
    mechanisms: Sequence[FuzzyMechanism]
    interaction_cost: float = 0.0

    def used_atoms(self, threshold: float = 1e-12) -> set[str]:
        return used_atoms(self.mechanisms, threshold)

    def shared_description_length(self, threshold: float = 1e-12) -> float:
        return shared_description_length(self.atoms, self.mechanisms, self.interaction_cost, threshold)

    def independent_description_length(self, threshold: float = 1e-12) -> float:
        return independent_description_length(self.atoms, self.mechanisms, threshold)

    def shared_mdl_gain(self, threshold: float = 1e-12) -> float:
        return shared_mdl_gain(self.atoms, self.mechanisms, self.interaction_cost, threshold)

    def mean_mechanisticity(self) -> float:
        if not self.mechanisms:
            return 0.0
        return float(sum(mechanism.mechanisticity() for mechanism in self.mechanisms) / len(self.mechanisms))
