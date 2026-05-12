from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec


@dataclass(frozen=True)
class DependencyStatus:
    package: str
    import_name: str
    available: bool
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "package": self.package,
            "import_name": self.import_name,
            "available": self.available,
            "reason": self.reason,
        }


def check_optional_dependency(package: str, import_name: str | None = None) -> DependencyStatus:
    import_name = import_name or package
    available = find_spec(import_name) is not None
    reason = "installed" if available else f"optional dependency {package!r} is not installed"
    return DependencyStatus(package=package, import_name=import_name, available=available, reason=reason)
