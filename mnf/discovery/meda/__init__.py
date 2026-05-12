"""CPU-local MEDA components for synthetic mechanism ecologies."""

from mnf.discovery.meda.certificate_builder import certificate_for_pair, certificates_for_pairs
from mnf.discovery.meda.active_design import active_meda_design
from mnf.discovery.meda.atom_proposal import AtomProposalResult, propose_atoms
from mnf.discovery.meda.chart_proposal import ChartProposal, propose_cyclic_chart, propose_linear_chart
from mnf.discovery.meda.mechanism_factorization import MechanismFactor, factorize_pairwise_mechanisms
from mnf.discovery.meda.pipeline import MEDAConfig, MEDAResult, discover_meda
from mnf.discovery.meda.response_surface import ResponseSurface
from mnf.discovery.meda.shared_mdl_pruning import SharedMDLDecision, greedy_shared_mdl_prune, membership_overlap
from mnf.discovery.meda.sparse_anova import boolean_fourier_coefficients, sparse_anova_terms
from mnf.discovery.meda.structural_disambiguation import (
    StructuralDisambiguation,
    disambiguate_pair,
    disambiguate_pairs,
)

__all__ = [
    "MEDAConfig",
    "MEDAResult",
    "ResponseSurface",
    "SharedMDLDecision",
    "StructuralDisambiguation",
    "AtomProposalResult",
    "ChartProposal",
    "MechanismFactor",
    "active_meda_design",
    "boolean_fourier_coefficients",
    "certificate_for_pair",
    "certificates_for_pairs",
    "discover_meda",
    "disambiguate_pair",
    "disambiguate_pairs",
    "factorize_pairwise_mechanisms",
    "greedy_shared_mdl_prune",
    "membership_overlap",
    "propose_atoms",
    "propose_cyclic_chart",
    "propose_linear_chart",
    "sparse_anova_terms",
]
