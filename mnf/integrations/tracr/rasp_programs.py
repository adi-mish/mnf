from __future__ import annotations


SUPPORTED_RASP_PROGRAMS = (
    "reverse",
    "histogram",
    "token_frequency",
    "sort",
    "balanced_parentheses",
    "modular_arithmetic",
)


def rasp_program_registry() -> dict[str, dict[str, object]]:
    return {
        name: {
            "name": name,
            "available": False,
            "reason": "Tracr package is not available in this environment",
        }
        for name in SUPPORTED_RASP_PROGRAMS
    }
