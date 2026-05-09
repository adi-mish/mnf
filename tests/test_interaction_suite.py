import pytest

from mnf.experiments.run_interaction_suite import run


@pytest.mark.smoke
@pytest.mark.slow
def test_interaction_suite_smoke():
    out = run(seeds=(0, 1), coactivations=(0.01, 0.1), decoder_cosines=(0.2, 0.9), overlaps=(0.0, 1.0))
    assert out["redundant_paths"]["single_ablation_misses_pair"] is True
    assert out["gating"]["gate_m_to_n"] == 1.0
    assert out["higher_order"]["third_order_effect"] == 1.0
    assert out["interaction_recovery"]["recovery"]["f1"] == 1.0
    assert out["noisy_interaction_recovery"]["rows"][0]["all_correct_rate"] == 1.0
    assert out["active_design"]["rows"][0]["mean_measurements"] == 22.0
    assert out["shared_atom_reuse"]["shared_mdl_gain"] > 0.0
    assert out["max_capacity_competition"] > 0.0
    assert out["phase_diagram_rows"] > 0
    assert out["mean_developmental_lag_steps"] > 0.0
