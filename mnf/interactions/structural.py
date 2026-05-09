from __future__ import annotations

from dataclasses import dataclass

from mnf.interactions.factorial_effects import FactorialEffects, classify_pairwise_interaction


@dataclass(frozen=True)
class StructuralInteractionEvidence:
    """Separates output-table interaction facts from structural claims."""

    phenomenological_label: str
    structural_label: str
    orientation: str = "unoriented"
    internal_gate_m_to_n: float = 0.0
    internal_gate_n_to_m: float = 0.0
    ambiguity_reason: str = ""

    @property
    def is_ambiguous(self) -> bool:
        return self.structural_label.startswith("ambiguous")

    def as_dict(self) -> dict[str, float | str | bool]:
        return {
            "phenomenological_label": self.phenomenological_label,
            "structural_label": self.structural_label,
            "orientation": self.orientation,
            "internal_gate_m_to_n": float(self.internal_gate_m_to_n),
            "internal_gate_n_to_m": float(self.internal_gate_n_to_m),
            "ambiguity_reason": self.ambiguity_reason,
            "is_ambiguous": self.is_ambiguous,
        }


def structural_interaction_evidence(
    effects: FactorialEffects,
    internal_gate_m_to_n: float | None = None,
    internal_gate_n_to_m: float | None = None,
    tol: float = 1e-6,
) -> StructuralInteractionEvidence:
    """Infer a structural label only when internal evidence orients it.

    Pairwise factorial output tables identify phenomenological interaction.
    Directed structural claims need activation-local or internal-variable
    evidence. Without such evidence, synergistic/gated output tables remain
    ambiguous.
    """

    label = classify_pairwise_interaction(effects, tol=tol)
    m_to_n = float(internal_gate_m_to_n or 0.0)
    n_to_m = float(internal_gate_n_to_m or 0.0)

    if label != "synergistic_or_gated":
        return StructuralInteractionEvidence(
            phenomenological_label=label,
            structural_label=label,
            internal_gate_m_to_n=m_to_n,
            internal_gate_n_to_m=n_to_m,
        )

    m_strong = abs(m_to_n) > tol
    n_strong = abs(n_to_m) > tol
    if m_strong and not n_strong:
        return StructuralInteractionEvidence(
            phenomenological_label=label,
            structural_label="directed_gate",
            orientation="m_to_n",
            internal_gate_m_to_n=m_to_n,
            internal_gate_n_to_m=n_to_m,
        )
    if n_strong and not m_strong:
        return StructuralInteractionEvidence(
            phenomenological_label=label,
            structural_label="directed_gate",
            orientation="n_to_m",
            internal_gate_m_to_n=m_to_n,
            internal_gate_n_to_m=n_to_m,
        )
    if m_strong and n_strong:
        return StructuralInteractionEvidence(
            phenomenological_label=label,
            structural_label="symmetric_synergy",
            orientation="bidirectional",
            internal_gate_m_to_n=m_to_n,
            internal_gate_n_to_m=n_to_m,
        )
    return StructuralInteractionEvidence(
        phenomenological_label=label,
        structural_label="ambiguous_without_internal_evidence",
        internal_gate_m_to_n=m_to_n,
        internal_gate_n_to_m=n_to_m,
        ambiguity_reason="same output factorial table can arise from directed gates, symmetric synergy, or saturation",
    )
