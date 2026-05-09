from mnf.atlas import MechanismChart, affine_gauge, gluing_error, identity_gauge
from mnf.benchmarks.atlas import no_global_chart_metrics


def test_affine_gauge_can_glue_shifted_charts():
    source = MechanismChart("source", ("z",), read=lambda sample: {"z": float(sample["x"])})
    target = MechanismChart("target", ("z",), read=lambda sample: {"z": 2.0 * float(sample["x"]) + 1.0})
    samples = [{"x": 0.0}, {"x": 1.0}, {"x": 2.0}]

    identity = gluing_error(source, target, identity_gauge(), samples)
    affine = gluing_error(source, target, affine_gauge(scale=2.0, offset=1.0), samples)

    assert identity.error > 0.0
    assert affine.error == 0.0


def test_no_global_chart_requires_context_indexed_gauge():
    metrics = no_global_chart_metrics()

    assert metrics["best_global_glue_error"] > 0.0
    assert metrics["context_indexed_glue_error"] == 0.0
    assert metrics["requires_context_indexed_gauge"] is True
