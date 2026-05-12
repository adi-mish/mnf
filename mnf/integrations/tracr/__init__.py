"""Optional Tracr integration boundary.

The CPU repo can detect and report Tracr availability. Actual compiled RASP
program support is intentionally optional because Tracr is not a core package
dependency and may not be available on every Python version.
"""

from mnf.integrations.tracr.availability import tracr_status

__all__ = ["tracr_status"]
