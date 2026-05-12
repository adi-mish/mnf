from __future__ import annotations

from mnf.integrations.tracr.availability import tracr_status


SUPPORTED_RASP_PROGRAMS = (
    "reverse",
    "histogram",
    "token_frequency",
    "sort",
    "balanced_parentheses",
    "modular_arithmetic",
    "shared_subroutine",
    "redundant_subroutines",
    "context_gated_subroutine",
    "shortcut_plus_correct_algorithm",
    "cyclic_arithmetic_plus_lookup",
    "length",
    "map_increment",
)


def rasp_program_registry() -> dict[str, dict[str, object]]:
    status = tracr_status()
    return {
        name: {
            "name": name,
            "available": status.available,
            "reason": "installed" if status.available else status.reason,
        }
        for name in SUPPORTED_RASP_PROGRAMS
    }
