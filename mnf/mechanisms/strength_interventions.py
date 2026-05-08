from __future__ import annotations

from collections.abc import Mapping


def linear_strength_intervention(value: float, eta: float, baseline: float = 0.0) -> float:
    """Interpolate an atom value between baseline and intact value."""

    if not 0.0 <= eta <= 1.0:
        raise ValueError("eta must be in [0, 1]")
    return float(baseline + eta * (value - baseline))


def weighted_atom_strengths(atom_membership: Mapping[str, float], eta: float) -> dict[str, float]:
    """Turn a mechanism strength into per-atom intervention weights."""

    if not 0.0 <= eta <= 1.0:
        raise ValueError("eta must be in [0, 1]")
    return {name: float(eta * weight) for name, weight in atom_membership.items()}
