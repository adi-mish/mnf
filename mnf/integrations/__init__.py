"""Optional integration probes for external interpretability toolchains."""

from mnf.integrations.availability import DependencyStatus, check_optional_dependency

__all__ = ["DependencyStatus", "check_optional_dependency"]
