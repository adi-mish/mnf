from mnf.benchmarks.interactions.noisy_recovery import noisy_recovery_sweep, noisy_recovery_trial


def test_noisy_recovery_exact_at_zero_noise():
    trial = noisy_recovery_trial(noise=0.0, seed=0)
    assert trial["accuracy"] == 1.0
    assert trial["f1"] == 1.0
    assert trial["all_correct"] == 1.0
    assert trial["max_abs_cell_error"] == 0.0


def test_noisy_recovery_sweep_degrades_with_large_noise():
    out = noisy_recovery_sweep(noise_levels=(0.0, 0.01, 0.3), seeds=tuple(range(50)))
    clean, low_noise, noisy = out["rows"]
    assert clean["all_correct_rate"] == 1.0
    assert low_noise["all_correct_rate"] == 1.0
    assert noisy["mean_accuracy"] <= clean["mean_accuracy"]
    assert noisy["mean_max_abs_cell_error"] > clean["mean_max_abs_cell_error"]
    assert noisy["mean_classification_tolerance"] > clean["mean_classification_tolerance"]
