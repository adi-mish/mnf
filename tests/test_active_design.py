from mnf.benchmarks.interactions.active_design import (
    active_design_baseline_sweep,
    active_design_sweep,
    active_design_trial,
    repeated_design_trial,
)
from mnf.interactions import active_interaction_discovery
from mnf.benchmarks.interactions.recovery_suite import DEFAULT_RECOVERY_NAMES, interaction_recovery_behavior


def test_active_design_recovers_exact_suite_with_pairwise_minimum():
    trial = active_design_trial(noise=0.0, seed=0, max_measurements=64)
    assert trial["measurements"] == 22
    assert trial["unique_states"] == 22
    assert trial["all_stable"] is True
    assert trial["accuracy"] == 1.0
    assert trial["f1"] == 1.0


def test_active_design_allocates_more_measurements_under_noise():
    clean = active_design_trial(noise=0.0, seed=0, max_measurements=128)
    noisy = active_design_trial(noise=0.02, seed=0, max_measurements=128)
    assert noisy["measurements"] >= clean["measurements"]
    assert noisy["unique_states"] == clean["unique_states"]
    assert noisy["accuracy"] == 1.0


def test_active_design_sweep_smoke():
    out = active_design_sweep(noise_levels=(0.0, 0.02), seeds=tuple(range(5)), max_measurements=128)
    assert out["rows"][0]["mean_measurements"] == 22.0
    assert out["rows"][1]["mean_measurements"] >= out["rows"][0]["mean_measurements"]


def test_repeated_design_baselines_smoke():
    uniform = repeated_design_trial(noise=0.0, seed=0, budget=22, mode="uniform")
    random = repeated_design_trial(noise=0.0, seed=0, budget=22, mode="random")

    assert uniform["accuracy"] == 1.0
    assert uniform["all_stable"] is True
    assert random["unique_states"] <= 22


def test_active_design_baseline_sweep_smoke():
    out = active_design_baseline_sweep(noise_levels=(0.0,), budgets=(22, 64), seeds=(0, 1))

    assert len(out["rows"]) == 2
    assert out["rows"][0]["active_mean_accuracy"] == 1.0
    assert out["rows"][0]["uniform_mean_accuracy"] == 1.0


def test_active_interaction_discovery_reports_labels():
    out = active_interaction_discovery(DEFAULT_RECOVERY_NAMES, interaction_recovery_behavior, max_measurements=64)
    assert out["all_stable"] is True
    assert out["labels"]["gate::worker"] == "synergistic_or_gated"
    assert out["labels"]["redundant_left::redundant_right"] == "redundant"
