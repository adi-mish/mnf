from mnf.experiments.schema import validate_research_sweeps


def test_research_sweep_schema_catches_missing_keys():
    result = validate_research_sweeps({})
    assert not result.ok
    assert any("$.ground_truth_suite is required" == error for error in result.errors)
