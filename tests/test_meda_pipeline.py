from mnf.discovery.meda import (
    MEDAConfig,
    boolean_fourier_coefficients,
    discover_meda,
    factorize_pairwise_mechanisms,
    sparse_anova_terms,
)


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


def test_meda_sparse_anova_and_factorization():
    names = ("a", "b")
    observations = {
        (0, 0): 0.0,
        (1, 0): 0.0,
        (0, 1): 0.0,
        (1, 1): 1.0,
    }
    coeffs = boolean_fourier_coefficients(names, observations)
    terms = sparse_anova_terms(names, observations, threshold=0.1)
    result = discover_meda(names, lambda state: float(state["a"] and state["b"]))
    factors = factorize_pairwise_mechanisms(result.factorials)

    assert coeffs[("a", "b")] != 0.0
    assert ("a", "b") in terms
    assert factors[0].label == "synergistic_or_gated"
