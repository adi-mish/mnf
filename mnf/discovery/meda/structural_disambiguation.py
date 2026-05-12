from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from mnf.interactions.factorial_effects import FactorialEffects
from mnf.interactions.structural import StructuralInteractionEvidence, structural_interaction_evidence


@dataclass(frozen=True)
class StructuralDisambiguation:
    pair: tuple[str, str]
    evidence: StructuralInteractionEvidence
    candidate_structures: tuple[str, ...]
    missing_evidence: tuple[str, ...]

    @property
    def ambiguous(self) -> bool:
        return self.evidence.is_ambiguous

    def as_dict(self) -> dict[str, object]:
        out = self.evidence.as_dict()
        out.update(
            {
                "pair": list(self.pair),
                "ambiguous": self.ambiguous,
                "candidate_structures": list(self.candidate_structures),
                "missing_evidence": list(self.missing_evidence),
            }
        )
        return out


def disambiguate_pair(
    pair: tuple[str, str],
    effects: FactorialEffects,
    internal_gate_evidence: Mapping[tuple[str, str], float] | None = None,
    tol: float = 1e-6,
) -> StructuralDisambiguation:
    internal_gate_evidence = internal_gate_evidence or {}
    m_to_n = internal_gate_evidence.get(pair)
    n_to_m = internal_gate_evidence.get((pair[1], pair[0]))
    evidence = structural_interaction_evidence(
        effects,
        internal_gate_m_to_n=m_to_n,
        internal_gate_n_to_m=n_to_m,
        tol=tol,
    )
    if evidence.is_ambiguous:
        candidates = ("directed_gate", "symmetric_synergy", "output_saturation")
        missing = (
            f"internal_gate_target:{pair[0]}->{pair[1]}",
            f"internal_gate_target:{pair[1]}->{pair[0]}",
        )
    else:
        candidates = (evidence.structural_label,)
        missing = ()
    return StructuralDisambiguation(
        pair=pair,
        evidence=evidence,
        candidate_structures=candidates,
        missing_evidence=missing,
    )


def disambiguate_pairs(
    factorials: Mapping[tuple[str, str], FactorialEffects],
    internal_gate_evidence: Mapping[tuple[str, str], float] | None = None,
    tol: float = 1e-6,
) -> dict[tuple[str, str], StructuralDisambiguation]:
    return {
        pair: disambiguate_pair(pair, effects, internal_gate_evidence=internal_gate_evidence, tol=tol)
        for pair, effects in factorials.items()
    }
