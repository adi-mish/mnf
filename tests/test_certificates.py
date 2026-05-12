from mnf.certificates import (
    CertificateThresholds,
    ConfidenceInterval,
    IdentificationSet,
    MechanismCertificate,
    certificate_report,
    pareto_frontier,
    percentile_interval,
    certificate_pareto_svg,
    sorted_certificate_names,
)


def test_certificate_dominance_treats_effect_as_benefit():
    strong = MechanismCertificate(intervention_error=0.01, shared_description_length=2.0, effect=1.0)
    weak = MechanismCertificate(intervention_error=0.02, shared_description_length=2.0, effect=0.8)

    assert strong.dominates(weak)
    assert not weak.dominates(strong)


def test_pareto_front_keeps_tradeoffs():
    compact = MechanismCertificate(intervention_error=0.1, shared_description_length=1.0, effect=0.7)
    faithful = MechanismCertificate(intervention_error=0.01, shared_description_length=4.0, effect=0.7)
    dominated = MechanismCertificate(intervention_error=0.2, shared_description_length=5.0, effect=0.1)

    assert pareto_frontier((compact, faithful, dominated)) == (0, 1)


def test_threshold_report_rejects_specific_certificate_dimensions():
    certificates = {
        "good": MechanismCertificate(intervention_error=0.01, effect=0.5, uncertainty=0.02),
        "bad": MechanismCertificate(intervention_error=0.4, effect=0.1, uncertainty=0.5),
    }
    report = certificate_report(
        certificates,
        CertificateThresholds(max_intervention_error=0.1, max_uncertainty=0.1, min_effect=0.2),
    )

    assert report.accepted == ("good",)
    assert report.rejected["bad"] == ("intervention_error", "uncertainty", "effect")


def test_identification_set_is_certificate_dimension():
    ambiguous = MechanismCertificate(
        effect=1.0,
        identification=IdentificationSet(
            representatives=("split", "merged"),
            diameter=0.25,
            reason="output_only_indistinguishable",
        ),
    )
    report = certificate_report(
        {"ambiguous": ambiguous},
        CertificateThresholds(max_identification_diameter=0.1, min_effect=0.5),
    )

    assert "identification_diameter" in report.rejected["ambiguous"]
    assert ambiguous.as_dict()["identification"]["is_ambiguous"] is True


def test_lagrangian_ranking_is_explicitly_weighted():
    certificates = {
        "short": MechanismCertificate(shared_description_length=1.0, effect=0.2),
        "useful": MechanismCertificate(shared_description_length=3.0, effect=1.0),
    }

    assert sorted_certificate_names(certificates, {"effect": 5.0})[0] == "useful"


def test_confidence_interval_width_and_percentiles():
    interval = percentile_interval([0.0, 1.0, 2.0, 3.0], confidence=0.5)
    explicit = ConfidenceInterval(low=1.0, high=2.0)

    assert interval.low <= interval.high
    assert explicit.width == 1.0
    assert explicit.contains(1.5)


def test_certificate_pareto_svg_writes_plot(tmp_path):
    report = {
        "pareto_indices": [0],
        "certificates": {
            "frontier": MechanismCertificate(intervention_error=0.01, shared_description_length=2.0).as_dict(),
            "dominated": MechanismCertificate(intervention_error=0.1, shared_description_length=4.0).as_dict(),
        },
    }
    path = tmp_path / "pareto.svg"

    certificate_pareto_svg(report, path)

    assert path.read_text().startswith("<svg")
