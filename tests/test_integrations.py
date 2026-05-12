from mnf.integrations import check_optional_dependency
from mnf.integrations.hf_transformers import hf_transformers_status, run_tiny_causal_lm_smoke
from mnf.integrations.tracr import default_tracr_hook_requests, rasp_program_registry, tracr_benchmark_cards, tracr_status
from mnf.integrations.transformer_lens import transformer_lens_status


def test_optional_integration_status_is_jsonable():
    status = check_optional_dependency("definitely-not-installed-mnf-package")
    assert status.available is False
    assert status.as_dict()["available"] is False


def test_external_interpretability_integrations_report_boundary():
    for status in (tracr_status(), transformer_lens_status()):
        data = status.as_dict()
        assert "package" in data
        assert "available" in data
        assert "reason" in data


def test_hf_transformers_smoke_boundary_is_jsonable():
    status = hf_transformers_status()
    assert "available" in status.as_dict()
    result = run_tiny_causal_lm_smoke(local_files_only=True)
    assert "available" in result.as_dict()


def test_tracr_optional_scaffold_has_named_programs_and_cards():
    registry = rasp_program_registry()
    cards = tracr_benchmark_cards()
    hooks = default_tracr_hook_requests()

    assert "reverse" in registry
    assert cards[0].name == "tracr_interacting_programs"
    assert hooks[0].site == "residual_stream"
