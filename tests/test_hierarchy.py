from mnf.benchmarks.hierarchy import make_hierarchy_dataset, compare_flat_vs_hierarchical_codes
from mnf.charts.hierarchical import absorption_boundary


def test_hierarchy_absorption_demo():
    ds = make_hierarchy_dataset(n=500, seed=0)
    metrics = compare_flat_vs_hierarchical_codes(ds)
    assert metrics["absorption_score"] > 0.2
    assert metrics["hierarchical_parent_error"] <= metrics["flat_parent_error"]
    assert absorption_boundary(child_rate=0.2, sparsity_penalty=1.0, recon_cost_absorbed=0.1, recon_cost_faithful=0.05)
