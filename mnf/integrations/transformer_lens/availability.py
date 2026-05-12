from __future__ import annotations

from mnf.integrations.availability import DependencyStatus, check_optional_dependency


def transformer_lens_status() -> DependencyStatus:
    return check_optional_dependency("transformer-lens", "transformer_lens")
