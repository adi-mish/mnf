from mnf.baselines import compare_feature_baselines


def test_feature_baselines_separate_labelability_from_causal_use():
    out = compare_feature_baselines(seed=0)

    assert out["random_labelable"]["mean_label_accuracy"] > 0.9
    assert out["trained_used"]["mean_label_accuracy"] > 0.9
    assert out["random_labelable"]["mean_causal_use_score"] < 0.2
    assert out["trained_used"]["mean_causal_use_score"] > 0.8
    assert out["random_labelable"]["false_mechanism_rate"] > 0.0
    assert out["trained_used"]["false_mechanism_rate"] == 0.0
