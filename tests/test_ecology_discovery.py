from mnf.discovery import mechanism_ecology_discovery, mechanism_ecology_discovery_from_joint_behavior


def test_mechanism_ecology_discovery_recovers_redundancy():
    matrix = mechanism_ecology_discovery(
        names=("m", "n"),
        behavior_by_pair={("m", "n"): lambda m_on, n_on: float(m_on or n_on)},
        memberships=({"shared": 1.0, "m_state": 1.0}, {"shared": 1.0, "n_state": 1.0}),
    )
    assert matrix.redundancy[0, 1] == 1.0
    assert matrix.overlap[0, 1] > 0.0


def test_mechanism_ecology_discovery_from_joint_behavior_recovers_gate():
    matrix = mechanism_ecology_discovery_from_joint_behavior(
        names=("gate", "worker", "context"),
        behavior_fn=lambda state: float(state["gate"] and state["worker"]) + 0.1 * float(state["context"]),
        memberships=({"gate_state": 1.0}, {"gate_state": 0.5, "worker": 1.0}, {"context": 1.0}),
    )
    assert matrix.gate[0, 1] == 1.0
    assert matrix.behavioral_synergy[0, 1] == 1.0
