from mnf.mechanisms import Atom, FuzzyMechanism, MechanismEcosystem, weighted_atom_strengths


def test_mechanisticity_penalizes_error_and_complexity():
    clean = FuzzyMechanism(
        name="clean",
        atom_membership={"a": 1.0},
        effect=1.0,
        intervention_error=0.0,
        invariance_error=0.0,
        naturalness_cost=0.0,
        description_length=1.0,
    )
    brittle = FuzzyMechanism(
        name="brittle",
        atom_membership={"a": 1.0},
        effect=1.0,
        intervention_error=0.5,
        invariance_error=0.5,
        naturalness_cost=0.5,
        description_length=10.0,
    )
    assert clean.mechanisticity() > brittle.mechanisticity()


def test_shared_mdl_rewards_reused_atoms():
    atoms = {
        "shared": Atom("shared", description_length=3.0),
        "a": Atom("a", description_length=1.0),
        "b": Atom("b", description_length=1.0),
    }
    m1 = FuzzyMechanism("m1", {"shared": 1.0, "a": 1.0}, effect=0.8, description_length=1.0)
    m2 = FuzzyMechanism("m2", {"shared": 1.0, "b": 1.0}, effect=0.8, description_length=1.0)
    ecosystem = MechanismEcosystem(atoms, (m1, m2), interaction_cost=0.25)
    assert ecosystem.shared_mdl_gain() == 2.75
    assert ecosystem.shared_description_length() < ecosystem.independent_description_length()


def test_strength_intervention_weights_atom_membership():
    weights = weighted_atom_strengths({"a": 1.0, "b": 0.25}, eta=0.4)
    assert weights == {"a": 0.4, "b": 0.1}
