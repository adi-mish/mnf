from mnf.discovery import mechanism_ecology_discovery


def test_mechanism_ecology_discovery_recovers_redundancy():
    matrix = mechanism_ecology_discovery(
        names=("m", "n"),
        behavior_by_pair={("m", "n"): lambda m_on, n_on: float(m_on or n_on)},
        memberships=({"shared": 1.0, "m_state": 1.0}, {"shared": 1.0, "n_state": 1.0}),
    )
    assert matrix.redundancy[0, 1] == 1.0
    assert matrix.overlap[0, 1] > 0.0
