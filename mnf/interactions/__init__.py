"""Interaction calculus for Interactive Mechanistic Normal Forms."""

from mnf.interactions.capacity import (
    capacity_competition,
    capacity_competition_from_matrices,
    decoder_gram_squared,
)
from mnf.interactions.dynamics import CoupledDynamicsParams, CoupledDynamicsTrace, first_crossing_step, simulate_coupled_dynamics
from mnf.interactions.factorial_effects import FactorialEffects, classify_pairwise_interaction, factorial_from_callable
from mnf.interactions.gating import conditional_gate_strength, gate_m_to_n_from_factorial, gate_n_to_m_from_factorial
from mnf.interactions.gradient_coupling import gradient_coupling, pairwise_gradient_coupling
from mnf.interactions.matrix import InteractionMatrix, build_interaction_matrix
from mnf.interactions.mediation import normalized_support_delta, support_delta
from mnf.interactions.overlap import pairwise_overlap_matrix, soft_jaccard, weighted_membership_dot
from mnf.interactions.redundancy import compensation_from_factorial, redundancy_from_factorial, single_ablation_misses_pair
from mnf.interactions.shapley import pairwise_shapley_values

__all__ = [
    "FactorialEffects",
    "classify_pairwise_interaction",
    "factorial_from_callable",
    "soft_jaccard",
    "weighted_membership_dot",
    "pairwise_overlap_matrix",
    "capacity_competition",
    "capacity_competition_from_matrices",
    "decoder_gram_squared",
    "conditional_gate_strength",
    "gate_m_to_n_from_factorial",
    "gate_n_to_m_from_factorial",
    "gradient_coupling",
    "pairwise_gradient_coupling",
    "support_delta",
    "normalized_support_delta",
    "redundancy_from_factorial",
    "compensation_from_factorial",
    "single_ablation_misses_pair",
    "pairwise_shapley_values",
    "CoupledDynamicsParams",
    "CoupledDynamicsTrace",
    "simulate_coupled_dynamics",
    "first_crossing_step",
    "InteractionMatrix",
    "build_interaction_matrix",
]
