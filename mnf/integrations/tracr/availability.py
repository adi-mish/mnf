from __future__ import annotations

from mnf.integrations.availability import DependencyStatus, check_optional_dependency


def tracr_status() -> DependencyStatus:
    return check_optional_dependency("tracr", "tracr")
