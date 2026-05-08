from mnf.benchmarks.random_control import random_vs_trained_control


def test_random_control_separates_labelability_from_use():
    out = random_vs_trained_control(n=800, seed=0)
    assert out["random_labelability"] > 0.75
    assert out["trained_labelability"] > 0.75
    assert out["trained_causal_use"] > out["random_causal_use"] + 0.5
