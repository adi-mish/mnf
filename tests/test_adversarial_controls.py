from mnf.benchmarks.memorization import memorization_control
from mnf.benchmarks.shortcut import make_shortcut_dataset, score_shortcut_candidates, shortcut_control
from mnf.experiments.run_adversarial_controls import run


def test_shortcut_control_rejects_spurious_feature_under_shift():
    ds = make_shortcut_dataset(n_per_environment=800, seed=0)
    scores = score_shortcut_candidates(ds)
    assert scores["causal"].accepted
    assert not scores["shortcut"].accepted
    assert scores["shortcut"].train_accuracy > 0.9
    assert scores["shortcut"].shifted_accuracy < 0.25
    assert scores["causal"].invariance_gap < 0.05


def test_memorization_control_has_high_fit_but_poor_generalization():
    out = memorization_control(n_train=256, n_test=256, seed=0)
    assert out["train_accuracy"] == 1.0
    assert 0.35 < out["test_accuracy"] < 0.65
    assert out["description_length"] == 256.0


def test_adversarial_controls_experiment_smoke():
    out = run(seed=0)
    shortcut = out["shortcut_spurious_correlation"]
    assert shortcut["causal_accepted"] == 1.0
    assert shortcut["shortcut_accepted"] == 0.0
    assert "memorizing_alignment" in out
