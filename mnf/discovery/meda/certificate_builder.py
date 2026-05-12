from __future__ import annotations

from collections.abc import Mapping

from mnf.certificates import IdentificationSet, MechanismCertificate
from mnf.discovery.meda.structural_disambiguation import StructuralDisambiguation
from mnf.interactions.factorial_effects import FactorialEffects


def certificate_for_pair(
    pair: tuple[str, str],
    effects: FactorialEffects,
    structural: StructuralDisambiguation,
    uncertainty: float = 0.0,
) -> MechanismCertificate:
    effect = max(abs(effects.joint_effect), abs(effects.synergy), effects.redundancy_score, effects.competition_score)
    if structural.ambiguous:
        identification = IdentificationSet(
            representatives=structural.candidate_structures,
            diameter=1.0,
            reason="structural_aliasing_under_output_table",
            distinguishable_by=structural.missing_evidence,
            indistinguishable_under=("pairwise_output_factorial_table",),
            metadata={"pair": pair},
        )
    else:
        identification = IdentificationSet(
            representatives=(structural.evidence.structural_label,),
            diameter=0.0,
            reason="oriented_by_available_evidence",
            distinguishable_by=(),
            indistinguishable_under=(),
            metadata={"pair": pair},
        )
    return MechanismCertificate(
        intervention_error=0.0,
        closure_error=0.0,
        shared_description_length=1.0,
        effect=float(effect),
        uncertainty=float(uncertainty),
        identification=identification,
    )


def certificates_for_pairs(
    factorials: Mapping[tuple[str, str], FactorialEffects],
    structural: Mapping[tuple[str, str], StructuralDisambiguation],
    uncertainty: float = 0.0,
) -> dict[str, MechanismCertificate]:
    return {
        f"{pair[0]}::{pair[1]}": certificate_for_pair(pair, effects, structural[pair], uncertainty=uncertainty)
        for pair, effects in factorials.items()
    }
