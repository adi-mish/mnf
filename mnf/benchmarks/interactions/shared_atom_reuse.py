from __future__ import annotations

from mnf.mechanisms import Atom, FuzzyMechanism, MechanismEcosystem


def make_shared_atom_ecosystem() -> MechanismEcosystem:
    atoms = {
        "route": Atom("route", kind="route", description_length=2.0),
        "left_state": Atom("left_state", kind="state", description_length=1.0),
        "right_state": Atom("right_state", kind="state", description_length=1.0),
    }
    left = FuzzyMechanism(
        name="left_lookup",
        atom_membership={"route": 1.0, "left_state": 1.0},
        effect=0.9,
        intervention_error=0.02,
        invariance_error=0.01,
        description_length=1.5,
    )
    right = FuzzyMechanism(
        name="right_lookup",
        atom_membership={"route": 1.0, "right_state": 1.0},
        effect=0.85,
        intervention_error=0.02,
        invariance_error=0.01,
        description_length=1.5,
    )
    return MechanismEcosystem(atoms=atoms, mechanisms=(left, right), interaction_cost=0.25)


def shared_atom_reuse_metrics() -> dict[str, float]:
    ecosystem = make_shared_atom_ecosystem()
    m0, m1 = ecosystem.mechanisms
    return {
        "atom_overlap": m0.atom_overlap(m1),
        "shared_description_length": ecosystem.shared_description_length(),
        "independent_description_length": ecosystem.independent_description_length(),
        "shared_mdl_gain": ecosystem.shared_mdl_gain(),
        "mean_mechanisticity": ecosystem.mean_mechanisticity(),
    }
