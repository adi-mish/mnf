from mnf.benchmarks.training_dynamics import make_training_emergence_trace, emergence_lead_metrics
from mnf.experiments.run_training_emergence import run


def test_training_emergence_mechanism_leads_behavior():
    trace = make_training_emergence_trace(seed=0)
    metrics = emergence_lead_metrics(trace)
    assert metrics["mechanism_lead_steps"] > 0
    assert metrics["final_intervention_score"] > 0.9
    assert metrics["final_behavioral_accuracy"] > 0.9


def test_training_emergence_experiment_smoke():
    out = run(seeds=(0, 1, 2))
    assert out["mean_mechanism_lead_steps"] > 5
    assert out["mean_correlation"] > 0.75
