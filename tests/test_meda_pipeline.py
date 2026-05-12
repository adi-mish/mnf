from mnf.discovery.meda import MEDAConfig, discover_meda


def test_meda_reports_structural_ambiguity_without_internal_evidence():
    result = discover_meda(
        names=("m", "n"),
        behavior_fn=lambda state: float(state["m"] and state["n"]),
        memberships=({"shared": 1.0, "m": 1.0}, {"shared": 1.0, "n": 1.0}),
    )

    structural = result.structural[("m", "n")]
    cert = result.certificates["m::n"]

    assert result.interaction_matrix.behavioral_synergy[0, 1] == 1.0
    assert structural.ambiguous is True
    assert "internal_gate_target:m->n" in structural.missing_evidence
    assert cert.identification.is_ambiguous is True
    assert result.shared_mdl is not None
    assert result.shared_mdl.gain > 0.0


def test_meda_orients_gate_with_internal_evidence():
    result = discover_meda(
        names=("gate", "worker"),
        behavior_fn=lambda state: float(state["gate"] and state["worker"]),
        internal_gate_evidence={("gate", "worker"): 1.0},
        config=MEDAConfig(structural_tolerance=1e-6),
    )

    structural = result.structural[("gate", "worker")]

    assert structural.ambiguous is False
    assert structural.evidence.structural_label == "directed_gate"
    assert structural.evidence.orientation == "m_to_n"
