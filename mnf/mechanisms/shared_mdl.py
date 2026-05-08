from __future__ import annotations

from collections.abc import Mapping, Sequence

from mnf.mechanisms.fuzzy import Atom, FuzzyMechanism


def used_atoms(mechanisms: Sequence[FuzzyMechanism], threshold: float = 1e-12) -> set[str]:
    out: set[str] = set()
    for mechanism in mechanisms:
        out |= mechanism.active_atoms(threshold)
    return out


def shared_description_length(
    atoms: Mapping[str, Atom],
    mechanisms: Sequence[FuzzyMechanism],
    interaction_cost: float = 0.0,
    threshold: float = 1e-12,
) -> float:
    atom_cost = sum(atoms[name].description_length for name in used_atoms(mechanisms, threshold) if name in atoms)
    mechanism_cost = sum(mechanism.description_length for mechanism in mechanisms)
    return float(atom_cost + mechanism_cost + interaction_cost)


def independent_description_length(
    atoms: Mapping[str, Atom],
    mechanisms: Sequence[FuzzyMechanism],
    threshold: float = 1e-12,
) -> float:
    total = 0.0
    for mechanism in mechanisms:
        total += mechanism.description_length
        for name in mechanism.active_atoms(threshold):
            if name in atoms:
                total += atoms[name].description_length
    return float(total)


def shared_mdl_gain(
    atoms: Mapping[str, Atom],
    mechanisms: Sequence[FuzzyMechanism],
    interaction_cost: float = 0.0,
    threshold: float = 1e-12,
) -> float:
    return independent_description_length(atoms, mechanisms, threshold) - shared_description_length(
        atoms, mechanisms, interaction_cost, threshold
    )
