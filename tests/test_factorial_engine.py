import numpy as np

from mnf.interactions import (
    FactorialEffects,
    bootstrap_factorial_effects,
    choose_next_pair_by_uncertainty,
    evaluate_design,
    full_factorial_design,
    higher_order_effect_from_observations,
    noisy_factorial_samples,
    pairwise_effects_from_observations,
    pairwise_factorial_design,
    sparse_higher_order_design,
)


def test_factorial_design_sizes():
    names = ("a", "b", "c", "d")
    assert len(full_factorial_design(len(names))) == 16
    assert len(pairwise_factorial_design(names)) == 11
    assert len(sparse_higher_order_design(names, max_order=2)) == 11


def test_pairwise_design_estimates_factorial_effects():
    names = ("gate", "worker", "context")
    design = pairwise_factorial_design(names)

    def behavior(state):
        return float(state["gate"] and state["worker"]) + 0.25 * float(state["context"])

    observations = evaluate_design(names, behavior, design)
    effects = pairwise_effects_from_observations(names, observations)
    assert effects[("gate", "worker")].gate_m_to_n == 1.0
    assert effects[("gate", "worker")].synergy == 1.0


def test_active_pair_selection_prefers_unobserved_then_uncertain():
    names = ("a", "b", "c")
    assert choose_next_pair_by_uncertainty(names, {}, observed_pairs=set()) == ("a", "b")
    pair = choose_next_pair_by_uncertainty(
        names,
        {("a", "b"): 0.1, ("a", "c"): 0.9, ("b", "c"): 0.3},
        observed_pairs={("a", "b"), ("a", "c"), ("b", "c")},
    )
    assert pair == ("a", "c")


def test_bootstrap_factorial_intervals_contain_signal():
    effects = FactorialEffects(y00=0.0, y10=0.0, y01=0.0, y11=1.0)
    samples = noisy_factorial_samples(effects, n=512, noise=0.02, seed=0)
    intervals = bootstrap_factorial_effects(samples, n_boot=200, seed=0)
    assert intervals["synergy"].low > 0.9
    assert np.isclose(intervals["gate_m_to_n"].mean, 1.0, atol=0.02)


def test_higher_order_effect_from_sparse_design():
    names = ("a", "b", "c")
    design = sparse_higher_order_design(names, max_order=3)

    def behavior(state):
        return float(state["a"] and state["b"] and state["c"])

    observations = evaluate_design(names, behavior, design)
    assert higher_order_effect_from_observations(names, observations, names) == 1.0
