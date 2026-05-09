from __future__ import annotations

import json

from mnf.certificates import (
    CertificateThresholds,
    IdentificationSet,
    MechanismCertificate,
    certificate_report,
    sorted_certificate_names,
)


def run() -> dict[str, object]:
    certificates = {
        "compact_but_weak": MechanismCertificate(
            intervention_error=0.08,
            glue_error=0.02,
            shared_description_length=1.0,
            effect=0.45,
            uncertainty=0.04,
            identification=IdentificationSet(
                representatives=("compact_factor",),
                diameter=0.0,
                reason="point_identified",
            ),
        ),
        "faithful_but_long": MechanismCertificate(
            intervention_error=0.01,
            glue_error=0.01,
            shared_description_length=4.0,
            effect=0.45,
            uncertainty=0.02,
            identification=IdentificationSet(
                representatives=("faithful_route_a", "faithful_route_b"),
                diameter=0.03,
                reason="atom_split_indistinguishable_under_output_only",
                distinguishable_by=("internal_route_marker",),
                indistinguishable_under=("output_only_interventions",),
            ),
        ),
        "dominated": MechanismCertificate(
            intervention_error=0.2,
            glue_error=0.1,
            shared_description_length=5.0,
            effect=0.1,
            uncertainty=0.2,
            identification=IdentificationSet(
                representatives=("dominated_a", "dominated_b", "dominated_c"),
                diameter=0.5,
                reason="too_broad_identification_set",
            ),
        ),
    }
    thresholds = CertificateThresholds(
        max_intervention_error=0.1,
        max_glue_error=0.05,
        max_uncertainty=0.1,
        max_identification_diameter=0.1,
        min_effect=0.2,
    )
    report = certificate_report(certificates, thresholds)
    return {
        "report": report.as_dict(),
        "lagrangian_rank_effect_weighted": list(sorted_certificate_names(certificates, {"effect": 5.0})),
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
