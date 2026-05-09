from mnf.benchmarks.interactions.higher_order import higher_order_interaction_metrics, triple_gate_behavior
from mnf.interactions import evaluate_design, search_higher_order_interactions, sparse_higher_order_design


def test_pairwise_only_design_abstains_on_unobserved_triple():
    names = ("gate_a", "gate_b", "worker")
    observations = evaluate_design(names, triple_gate_behavior, sparse_higher_order_design(names, max_order=2))
    search = search_higher_order_interactions(names, observations, min_order=3, max_order=3)
    assert search[0].label == "unobserved"
    assert search[0].missing_cells > 0


def test_triple_design_recovers_higher_order_gate():
    names = ("gate_a", "gate_b", "worker")
    observations = evaluate_design(names, triple_gate_behavior, sparse_higher_order_design(names, max_order=3))
    search = search_higher_order_interactions(names, observations, min_order=3, max_order=3)
    assert search[0].label == "positive_higher_order"
    assert search[0].effect == 1.0


def test_higher_order_metrics_include_k_way_sweep():
    metrics = higher_order_interaction_metrics()
    labels = [row["label"] for row in metrics["k_way_sweep"]["rows"]]
    assert labels == ["positive_higher_order", "positive_higher_order", "positive_higher_order"]
