from mnf.baselines import compare_dictionary_baselines


def test_dictionary_baselines_and_acdc_control_expose_failures():
    out = compare_dictionary_baselines(seed=0)

    assert out["random_labelable"]["mean_label_accuracy"] > 0.85
    assert out["random_labelable"]["mean_causal_use_score"] < 0.25
    assert out["acdc_redundancy_failure"]["misses_redundancy"] is True
    assert out["acdc_redundancy_failure"]["dual_drop"] == 1.0
