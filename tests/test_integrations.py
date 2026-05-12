from mnf.integrations import check_optional_dependency
from mnf.integrations.tracr import tracr_status
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
