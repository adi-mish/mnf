"""Optional Tracr integration boundary.

The CPU repo can detect and report Tracr availability. Actual compiled RASP
program support is intentionally optional because Tracr is not a core package
dependency and may not be available on every Python version.
"""

from mnf.integrations.tracr.availability import tracr_status
from mnf.integrations.tracr.load_tracr import load_tracr_module
from mnf.integrations.tracr.rasp_programs import SUPPORTED_RASP_PROGRAMS, rasp_program_registry
from mnf.integrations.tracr.tracr_activation_hooks import TracrHookRequest, default_tracr_hook_requests
from mnf.integrations.tracr.tracr_benchmark_cards import tracr_benchmark_cards

__all__ = [
    "SUPPORTED_RASP_PROGRAMS",
    "TracrHookRequest",
    "default_tracr_hook_requests",
    "load_tracr_module",
    "rasp_program_registry",
    "tracr_benchmark_cards",
    "tracr_status",
]
