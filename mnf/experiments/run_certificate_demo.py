from __future__ import annotations

import json

from mnf.certificates import (
    CertificateThresholds,
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
        ),
        "faithful_but_long": MechanismCertificate(
            intervention_error=0.01,
            glue_error=0.01,
            shared_description_length=4.0,
            effect=0.45,
            uncertainty=0.02,
        ),
        "dominated": MechanismCertificate(
            intervention_error=0.2,
            glue_error=0.1,
            shared_description_length=5.0,
            effect=0.1,
            uncertainty=0.2,
        ),
    }
    thresholds = CertificateThresholds(
        max_intervention_error=0.1,
        max_glue_error=0.05,
        max_uncertainty=0.1,
        min_effect=0.2,
    )
    report = certificate_report(certificates, thresholds)
    return {
        "report": report.as_dict(),
        "lagrangian_rank_effect_weighted": list(sorted_certificate_names(certificates, {"effect": 5.0})),
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
