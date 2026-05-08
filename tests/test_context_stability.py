from mnf.benchmarks.interactions.context_stability import context_labels, context_stability_metrics
from mnf.interactions import FactorialEffects, classify_pairwise_interaction_with_uncertainty


def test_context_shift_changes_focal_pair_label():
    labels_on = context_labels(context_on=True)
    labels_off = context_labels(context_on=False)

    assert labels_on[("left", "right")] == "additive"
    assert labels_off[("left", "right")] == "synergistic_or_gated"


def test_context_stability_metrics_report_changed_pairs():
    metrics = context_stability_metrics()

    assert metrics["n_changed_pairs"] >= 1
    assert metrics["stable_label_rate"] < 1.0
    assert metrics["focal_context_on_label"] == "additive"
    assert metrics["focal_context_off_label"] == "synergistic_or_gated"


def test_uncertain_classifier_abstains_near_margin():
    effects = FactorialEffects(y00=0.0, y10=1.0, y01=1.0, y11=2.15)

    assert classify_pairwise_interaction_with_uncertainty(effects, tol=0.1) == "uncertain"
    assert classify_pairwise_interaction_with_uncertainty(effects, tol=0.01) == "synergistic_or_gated"
