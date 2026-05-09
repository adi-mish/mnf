from __future__ import annotations

from dataclasses import dataclass

from mnf.certificates.certificate import MechanismCertificate


@dataclass(frozen=True)
class CertificateThresholds:
    max_obs_error: float = float("inf")
    max_intervention_error: float = float("inf")
    max_invariance_error: float = float("inf")
    max_glue_error: float = float("inf")
    max_naturalness_cost: float = float("inf")
    max_closure_error: float = float("inf")
    max_shared_description_length: float = float("inf")
    max_uncertainty: float = float("inf")
    min_effect: float = 0.0

    def accepts(self, certificate: MechanismCertificate) -> bool:
        return (
            certificate.obs_error <= self.max_obs_error
            and certificate.intervention_error <= self.max_intervention_error
            and certificate.invariance_error <= self.max_invariance_error
            and certificate.glue_error <= self.max_glue_error
            and certificate.naturalness_cost <= self.max_naturalness_cost
            and certificate.closure_error <= self.max_closure_error
            and certificate.shared_description_length <= self.max_shared_description_length
            and certificate.uncertainty <= self.max_uncertainty
            and certificate.effect >= self.min_effect
        )

    def rejection_reasons(self, certificate: MechanismCertificate) -> tuple[str, ...]:
        reasons: list[str] = []
        checks = (
            ("obs_error", certificate.obs_error, self.max_obs_error, "<="),
            ("intervention_error", certificate.intervention_error, self.max_intervention_error, "<="),
            ("invariance_error", certificate.invariance_error, self.max_invariance_error, "<="),
            ("glue_error", certificate.glue_error, self.max_glue_error, "<="),
            ("naturalness_cost", certificate.naturalness_cost, self.max_naturalness_cost, "<="),
            ("closure_error", certificate.closure_error, self.max_closure_error, "<="),
            (
                "shared_description_length",
                certificate.shared_description_length,
                self.max_shared_description_length,
                "<=",
            ),
            ("uncertainty", certificate.uncertainty, self.max_uncertainty, "<="),
        )
        for name, value, threshold, _ in checks:
            if value > threshold:
                reasons.append(name)
        if certificate.effect < self.min_effect:
            reasons.append("effect")
        return tuple(reasons)
