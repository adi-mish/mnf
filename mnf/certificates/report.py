from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from mnf.certificates.certificate import MechanismCertificate
from mnf.certificates.pareto import pareto_frontier, rank_by_lagrangian
from mnf.certificates.thresholds import CertificateThresholds


@dataclass(frozen=True)
class CertificateReport:
    names: tuple[str, ...]
    certificates: tuple[MechanismCertificate, ...]
    pareto_indices: tuple[int, ...]
    accepted: tuple[str, ...]
    rejected: Mapping[str, tuple[str, ...]]

    def as_dict(self) -> dict[str, object]:
        return {
            "names": list(self.names),
            "pareto_indices": list(self.pareto_indices),
            "accepted": list(self.accepted),
            "rejected": {name: list(reasons) for name, reasons in self.rejected.items()},
            "certificates": {
                name: certificate.as_dict()
                for name, certificate in zip(self.names, self.certificates)
            },
        }


def certificate_report(
    named_certificates: Mapping[str, MechanismCertificate],
    thresholds: CertificateThresholds | None = None,
) -> CertificateReport:
    thresholds = thresholds or CertificateThresholds()
    names = tuple(named_certificates)
    certificates = tuple(named_certificates.values())
    accepted = tuple(name for name, certificate in named_certificates.items() if thresholds.accepts(certificate))
    rejected = {
        name: thresholds.rejection_reasons(certificate)
        for name, certificate in named_certificates.items()
        if not thresholds.accepts(certificate)
    }
    return CertificateReport(
        names=names,
        certificates=certificates,
        pareto_indices=pareto_frontier(certificates),
        accepted=accepted,
        rejected=rejected,
    )


def sorted_certificate_names(
    named_certificates: Mapping[str, MechanismCertificate],
    weights: Mapping[str, float] | None = None,
) -> tuple[str, ...]:
    names = tuple(named_certificates)
    certificates = tuple(named_certificates.values())
    return tuple(names[i] for i in rank_by_lagrangian(certificates, weights))
