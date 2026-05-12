from mnf.baselines import compare_dictionary_baselines
from mnf.baselines.acdc_synthetic import acdc_redundancy_failure_demo
from mnf.baselines.causal_scrubbing_tiny import causal_scrubbing_residual
from mnf.baselines.ica import ica_first_direction
from mnf.baselines.kmeans_dictionary import kmeans_dictionary_direction
from mnf.baselines.linear_das import linear_das_direction


def test_dictionary_baselines_and_acdc_control_expose_failures():
    out = compare_dictionary_baselines(seed=0)

    assert out["random_labelable"]["mean_label_accuracy"] > 0.85
    assert out["random_labelable"]["mean_causal_use_score"] < 0.25
    assert out["acdc_redundancy_failure"]["misses_redundancy"] is True
    assert out["acdc_redundancy_failure"]["dual_drop"] == 1.0


def test_named_cpu_baseline_modules_are_importable():
    out = compare_dictionary_baselines(seed=1)
    row = out["trained_used"]["rows"][0]

    assert row["name"] in {"ica_first", "kmeans_dictionary"}
    assert acdc_redundancy_failure_demo()["misses_redundancy"] is True
    assert callable(ica_first_direction)
    assert callable(kmeans_dictionary_direction)
    assert callable(linear_das_direction)
    assert causal_scrubbing_residual(lambda s: 1.0, lambda s: 0.25, {}) == 0.75
