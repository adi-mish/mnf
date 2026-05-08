from mnf.experiments.run_absorption_phase import run


def test_absorption_phase_sweep_smoke():
    out = run(
        n=300,
        child_rates=(0.2,),
        parent_only_rates=(0.3,),
        noise_levels=(0.03,),
        seeds=(0, 1),
    )
    row = out["rows"][0]
    assert row["absorption_score"] > 0.2
    assert row["hierarchical_parent_error"] <= row["flat_parent_error"]
    assert row["hierarchical_error_reduction"] >= 0.0
