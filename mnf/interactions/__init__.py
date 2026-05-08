"""Interaction calculus for Interactive Mechanistic Normal Forms."""

from mnf.interactions.active import (
    ActivePairStatus,
    active_interaction_discovery,
    active_pair_statuses,
    choose_next_intervention_state,
    pair_cell_states,
    pair_status,
)
from mnf.interactions.capacity import (
    capacity_competition,
    capacity_competition_from_matrices,
    decoder_gram_squared,
)
from mnf.interactions.bounds import (
    all_required_contrasts_stable,
    contrast_error_bound,
    higher_order_contrast_error_bound,
    pairwise_contrast_error_bounds,
    sign_is_stable,
)
from mnf.interactions.design import (
    InterventionState,
    choose_next_pair_by_uncertainty,
    evaluate_design,
    full_factorial_design,
    higher_order_effect_from_observations,
    pairwise_effects_from_observations,
    pairwise_factorial_design,
    sparse_higher_order_design,
)
from mnf.interactions.dynamics import CoupledDynamicsParams, CoupledDynamicsTrace, first_crossing_step, simulate_coupled_dynamics
from mnf.interactions.factorial_effects import (
    FactorialEffects,
    classify_pairwise_interaction,
    classify_pairwise_interaction_with_uncertainty,
    factorial_from_callable,
)
from mnf.interactions.gating import conditional_gate_strength, gate_m_to_n_from_factorial, gate_n_to_m_from_factorial
from mnf.interactions.gradient_coupling import gradient_coupling, pairwise_gradient_coupling
from mnf.interactions.matrix import InteractionMatrix, build_interaction_matrix
from mnf.interactions.mediation import normalized_support_delta, support_delta
from mnf.interactions.overlap import pairwise_overlap_matrix, soft_jaccard, weighted_membership_dot
from mnf.interactions.recovery import InteractionRecoveryReport, edge_f1, label_recovery_report, threshold_edges
from mnf.interactions.redundancy import compensation_from_factorial, redundancy_from_factorial, single_ablation_misses_pair
from mnf.interactions.shapley import pairwise_shapley_values
from mnf.interactions.uncertainty import EffectInterval, bootstrap_factorial_effects, noisy_factorial_samples

__all__ = [
    "ActivePairStatus",
    "pair_cell_states",
    "pair_status",
    "active_pair_statuses",
    "choose_next_intervention_state",
    "active_interaction_discovery",
    "FactorialEffects",
    "classify_pairwise_interaction",
    "classify_pairwise_interaction_with_uncertainty",
    "factorial_from_callable",
    "soft_jaccard",
    "weighted_membership_dot",
    "pairwise_overlap_matrix",
    "capacity_competition",
    "capacity_competition_from_matrices",
    "decoder_gram_squared",
    "contrast_error_bound",
    "pairwise_contrast_error_bounds",
    "higher_order_contrast_error_bound",
    "sign_is_stable",
    "all_required_contrasts_stable",
    "InterventionState",
    "full_factorial_design",
    "pairwise_factorial_design",
    "sparse_higher_order_design",
    "evaluate_design",
    "pairwise_effects_from_observations",
    "higher_order_effect_from_observations",
    "choose_next_pair_by_uncertainty",
    "conditional_gate_strength",
    "gate_m_to_n_from_factorial",
    "gate_n_to_m_from_factorial",
    "gradient_coupling",
    "pairwise_gradient_coupling",
    "support_delta",
    "normalized_support_delta",
    "InteractionRecoveryReport",
    "threshold_edges",
    "edge_f1",
    "label_recovery_report",
    "redundancy_from_factorial",
    "compensation_from_factorial",
    "single_ablation_misses_pair",
    "pairwise_shapley_values",
    "EffectInterval",
    "bootstrap_factorial_effects",
    "noisy_factorial_samples",
    "CoupledDynamicsParams",
    "CoupledDynamicsTrace",
    "simulate_coupled_dynamics",
    "first_crossing_step",
    "InteractionMatrix",
    "build_interaction_matrix",
]
