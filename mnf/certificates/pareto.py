from __future__ import annotations

from collections.abc import Mapping, Sequence

from mnf.certificates.certificate import MechanismCertificate


def pareto_frontier(
    certificates: Sequence[MechanismCertificate],
    tolerances: Mapping[str, float] | None = None,
) -> tuple[int, ...]:
    """Return indices of certificates not dominated by any other certificate."""

    frontier: list[int] = []
    for i, certificate in enumerate(certificates):
        dominated = False
        for j, other in enumerate(certificates):
            if i == j:
                continue
            if other.dominates(certificate, tolerances=tolerances):
                dominated = True
                break
        if not dominated:
            frontier.append(i)
    return tuple(frontier)


def rank_by_lagrangian(
    certificates: Sequence[MechanismCertificate],
    weights: Mapping[str, float] | None = None,
) -> tuple[int, ...]:
    return tuple(sorted(range(len(certificates)), key=lambda i: certificates[i].to_lagrangian(weights)))
