from mnf.baselines.cyclic import compare_cyclic_baselines


def test_typed_cyclic_chart_beats_scalar_cyclic_baseline():
    out = compare_cyclic_baselines(n=900, noise=0.03, seed=0)
    assert out["typed_label_error"] < 0.05
    assert out["typed_rotation_error"] < 0.05
    assert out["scalar_label_error"] > 0.4
    assert out["scalar_rotation_error"] > 0.4
