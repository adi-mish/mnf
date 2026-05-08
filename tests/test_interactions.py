import numpy as np

from mnf.benchmarks.interactions import (
    capacity_competition_score,
    developmental_bootstrap_metrics,
    gating_metrics,
    mechanism_death_metrics,
    redundant_paths_metrics,
    synergistic_paths_metrics,
)
from mnf.benchmarks.interactions.phase_diagram import factorial_interaction_phase_diagram
from mnf.interactions import (
    FactorialEffects,
    build_interaction_matrix,
    gradient_coupling,
    pairwise_shapley_values,
    single_ablation_misses_pair,
    soft_jaccard,
)


def test_redundant_paths_require_factorial_ablation():
    metrics = redundant_paths_metrics()
    assert metrics["drop_m_from_joint"] == 0.0
    assert metrics["drop_n_from_joint"] == 0.0
    assert metrics["dual_ablation_drop"] == 1.0
    assert metrics["redundancy_score"] == 1.0
    assert metrics["single_ablation_misses_pair"] is True


def test_gating_has_conditional_effect_without_marginal_worker_effect():
    metrics = gating_metrics()
    assert metrics["worker_effect_when_gate_on"] == 1.0
    assert metrics["worker_effect_when_gate_off"] == 0.0
    assert metrics["gate_m_to_n"] == 1.0
    assert metrics["synergy"] == 1.0


def test_synergy_shapley_allocates_joint_gain():
    metrics = synergistic_paths_metrics()
    assert metrics["synergy"] == 1.0
    assert metrics["shapley_total"] == 1.0
    assert metrics["shapley_m"] == metrics["shapley_n"]


def test_single_ablation_miss_helper_rejects_and_gate():
    redundant = FactorialEffects(y00=0.0, y10=1.0, y01=1.0, y11=1.0)
    gated = FactorialEffects(y00=0.0, y10=0.0, y01=0.0, y11=1.0)
    assert single_ablation_misses_pair(redundant)
    assert not single_ablation_misses_pair(gated)


def test_soft_jaccard_and_interaction_matrix():
    assert soft_jaccard({"a": 1.0, "b": 0.5}, {"a": 0.5, "c": 1.0}) == 0.2
    matrix = build_interaction_matrix(
        ["m", "n"],
        factorials={("m", "n"): FactorialEffects(y00=0.0, y10=1.0, y01=1.0, y11=1.0)},
        memberships=[{"a": 1.0}, {"a": 0.5, "b": 0.5}],
    )
    assert matrix.redundancy[0, 1] == 1.0
    assert matrix.overlap[0, 1] > 0.0


def test_capacity_competition_monotone_in_coactivation_and_alignment():
    low = capacity_competition_score(coactivation=0.01, decoder_cosine=0.2, overlap=0.5)
    high_q = capacity_competition_score(coactivation=0.2, decoder_cosine=0.2, overlap=0.5)
    high_alignment = capacity_competition_score(coactivation=0.01, decoder_cosine=0.9, overlap=0.5)
    assert high_q > low
    assert high_alignment > low


def test_gradient_coupling_signs():
    assert gradient_coupling(np.array([1.0, 0.0]), np.array([2.0, 0.0])) > 0.99
    assert gradient_coupling(np.array([1.0, 0.0]), np.array([-1.0, 0.0])) < -0.99


def test_developmental_bootstrap_and_mechanism_death():
    bootstrap = developmental_bootstrap_metrics()
    death = mechanism_death_metrics()
    assert bootstrap["developmental_lag_steps"] > 0
    assert bootstrap["final_dependent_strength"] > 0.9
    assert death["suppression_delta"] > 0.1


def test_factorial_interaction_phase_diagram_spans_labels():
    out = factorial_interaction_phase_diagram(redundancy_weights=(0.0, 1.0), synergy_weights=(0.0, 1.0))
    labels = {row["interaction_label"] for row in out["rows"]}
    assert "redundant" in labels
    assert "synergistic_or_gated" in labels
