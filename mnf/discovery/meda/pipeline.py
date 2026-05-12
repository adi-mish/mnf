from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from mnf.certificates import MechanismCertificate
from mnf.discovery.meda.certificate_builder import certificates_for_pairs
from mnf.discovery.meda.response_surface import ResponseSurface
from mnf.discovery.meda.shared_mdl_pruning import SharedMDLDecision, greedy_shared_mdl_prune
from mnf.discovery.meda.structural_disambiguation import StructuralDisambiguation, disambiguate_pairs
from mnf.interactions import InteractionMatrix, build_interaction_matrix
from mnf.interactions.factorial_effects import FactorialEffects


@dataclass(frozen=True)
class MEDAConfig:
    max_order: int = 2
    full_factorial: bool = False
    context_on: bool = True
    structural_tolerance: float = 1e-6


@dataclass(frozen=True)
class MEDAResult:
    response_surface: ResponseSurface
    factorials: dict[tuple[str, str], FactorialEffects]
    interaction_matrix: InteractionMatrix
    structural: dict[tuple[str, str], StructuralDisambiguation]
    certificates: dict[str, MechanismCertificate]
    shared_mdl: SharedMDLDecision | None = None
    diagnostics: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "response_surface": self.response_surface.as_dict(),
            "interaction_matrix": self.interaction_matrix.as_dict(),
            "structural": {f"{a}::{b}": value.as_dict() for (a, b), value in self.structural.items()},
            "certificates": {name: certificate.as_dict() for name, certificate in self.certificates.items()},
            "shared_mdl": None if self.shared_mdl is None else self.shared_mdl.as_dict(),
            "diagnostics": dict(self.diagnostics),
        }


def discover_meda(
    names: Sequence[str],
    behavior_fn: Callable[[Mapping[str, bool]], float],
    memberships: Sequence[Mapping[str, float]] | None = None,
    internal_gate_evidence: Mapping[tuple[str, str], float] | None = None,
    atom_costs: Mapping[str, float] | None = None,
    config: MEDAConfig | None = None,
) -> MEDAResult:
    config = config or MEDAConfig()
    surface = ResponseSurface.from_callable(
        names,
        behavior_fn,
        max_order=config.max_order,
        full_factorial=config.full_factorial,
        context_on=config.context_on,
    )
    factorials = surface.pairwise_factorials()
    matrix = build_interaction_matrix(names=names, factorials=factorials, memberships=memberships)
    structural = disambiguate_pairs(
        factorials,
        internal_gate_evidence=internal_gate_evidence,
        tol=config.structural_tolerance,
    )
    certificates = certificates_for_pairs(factorials, structural)
    shared_mdl = None
    if memberships is not None:
        shared_mdl = greedy_shared_mdl_prune(memberships, atom_costs=atom_costs)
    return MEDAResult(
        response_surface=surface,
        factorials=factorials,
        interaction_matrix=matrix,
        structural=structural,
        certificates=certificates,
        shared_mdl=shared_mdl,
        diagnostics={
            "n_mechanisms": len(tuple(names)),
            "n_pairs": len(factorials),
            "n_ambiguous_pairs": sum(value.ambiguous for value in structural.values()),
        },
    )
