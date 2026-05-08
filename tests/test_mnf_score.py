from mnf.benchmarks.synthetic import make_chain_program, sample_chain_data, chain_intervention_dataset
from mnf.core.mechanism import MechanismExplanation


def test_mnf_score_low_for_true_program():
    prog = make_chain_program()
    inputs, states = sample_chain_data(64, seed=0)
    outputs = [s["y"] for s in states]
    mech = MechanismExplanation(
        program=prog,
        encoder=lambda low: low,
        decoder=lambda state: state["y"],
        parameters_description="ground_truth",
    )
    int_inputs, interventions, int_states = chain_intervention_dataset(16, seed=1)
    score = mech.score(inputs, outputs, beta_mdl=0.001)
    int_err = mech.intervention_error(int_inputs, interventions, int_states)
    assert score.obs_error < 1e-12
    assert int_err < 1e-12
    assert score.total < 0.1
