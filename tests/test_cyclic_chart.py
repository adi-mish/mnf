from mnf.benchmarks.cyclic import make_weekday_rotation_dataset, weekday_space
from mnf.charts.cyclic import CyclicChart


def test_cyclic_chart_recovers_weekday_rotation():
    ds = make_weekday_rotation_dataset(n=700, noise=0.01, seed=3)
    chart = CyclicChart.fit(ds.activations, ds.labels, weekday_space)
    pred = chart.encode_label(ds.activations)
    base_error = sum(a != b for a, b in zip(pred, ds.labels)) / len(ds.labels)
    rot_error = chart.intervention_error_after_rotation(ds.activations, ds.labels, ds.rotation_steps)
    assert base_error < 0.02
    assert rot_error < 0.05
