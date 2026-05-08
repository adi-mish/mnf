from mnf.experiments.run_cyclic_noise_sweep import run as run_cyclic_noise_sweep
from mnf.experiments.run_transition_atom_sweep import run as run_transition_atom_sweep


def test_transition_atom_sweep_smoke():
    out = run_transition_atom_sweep(
        n=500,
        gate_separations=(2.5,),
        noise_levels=(0.05,),
        seeds=(0, 1),
    )
    row = out["rows"][0]
    assert row["molt_to_global_mse_ratio"] < 0.2


def test_cyclic_noise_sweep_smoke():
    out = run_cyclic_noise_sweep(
        n=400,
        noise_levels=(0.03,),
        seeds=(0, 1),
    )
    row = out["rows"][0]
    assert row["base_label_error"] < 0.05
    assert row["rotation_intervention_error"] < 0.1
