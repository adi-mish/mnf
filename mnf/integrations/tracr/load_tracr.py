from __future__ import annotations

from typing import Any

from mnf.integrations.tracr.availability import tracr_status


def load_tracr_module() -> Any:
    status = tracr_status()
    if not status.available:
        raise ImportError(status.reason)
    import tracr  # type: ignore[import-not-found]

    return tracr
