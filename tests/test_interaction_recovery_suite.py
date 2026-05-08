from mnf.benchmarks.interactions.recovery_suite import interaction_recovery_suite


def test_interaction_recovery_suite_recovers_ground_truth_labels():
    out = interaction_recovery_suite(bootstrap_n=100, seed=0)
    assert out["design_size"] == 22
    assert out["n_pairs"] == 15
    assert out["recovery"]["f1"] == 1.0
    assert out["recovery"]["accuracy"] == 1.0
    assert out["predicted_label_counts"]["redundant"] == 1
    assert out["predicted_label_counts"]["synergistic_or_gated"] == 1
    assert out["predicted_label_counts"]["competitive"] == 1


def test_interaction_recovery_suite_bootstrap_confidence_is_informative():
    out = interaction_recovery_suite(bootstrap_n=100, seed=0)
    assert out["gate_synergy_ci"]["synergy"]["low"] > 0.8
    assert out["redundancy_ci"]["redundancy_score"]["low"] > 0.8
