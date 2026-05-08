from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import numpy as np


@dataclass(frozen=True)
class Intervention:
    """High-level intervention on a named causal variable."""

    variable: str
    kind: str
    value: Any = None

    def apply(self, current: Any) -> Any:
        if self.kind == "clamp":
            return self.value
        if self.kind == "ablate":
            return 0
        if self.kind == "shift":
            return current + self.value
        if self.kind == "scale":
            return current * self.value
        if self.kind == "rotate":
            # Rotation is chart-specific.  At the generic level, expect an
            # integer modulo period or an angle value already encoded in value.
            if isinstance(current, tuple) and len(current) == 2:
                idx, period = current
                return ((idx + int(self.value)) % period, period)
            return current + self.value
        raise ValueError(f"Unknown intervention kind: {self.kind}")


@dataclass(frozen=True)
class PatchIntervention:
    """Patch a variable from a source run into a target run."""

    variable: str
    source_state: Mapping[str, Any]

    def to_intervention(self) -> Intervention:
        return Intervention(self.variable, "clamp", self.source_state[self.variable])


def apply_interventions(state: Mapping[str, Any], interventions: Mapping[str, Intervention]) -> dict[str, Any]:
    out = dict(state)
    for name, intervention in interventions.items():
        if name in out:
            out[name] = intervention.apply(out[name])
    return out


def activation_patch(target: np.ndarray, source: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Patch selected activation coordinates from source into target."""
    target_arr = np.asarray(target, dtype=float).copy()
    source_arr = np.asarray(source, dtype=float)
    mask_arr = np.asarray(mask, dtype=bool)
    if target_arr.shape != source_arr.shape or target_arr.shape != mask_arr.shape:
        raise ValueError("target, source, and mask must have identical shapes")
    target_arr[mask_arr] = source_arr[mask_arr]
    return target_arr
