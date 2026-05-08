from mnf.experiments.run_ground_truth_suite import run


def test_ground_truth_suite_recovers_direct_edges():
    out = run(seed=0)
    rows = out["rows"]
    assert {row["benchmark"] for row in rows} == {
        "scalar_chain",
        "gated_xor",
        "modular_addition_C7",
        "relational_lookup",
    }
    assert all(row["f1"] == 1.0 for row in rows)
