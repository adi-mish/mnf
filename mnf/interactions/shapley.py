from __future__ import annotations

from mnf.interactions.factorial_effects import FactorialEffects


def pairwise_shapley_values(effects: FactorialEffects) -> dict[str, float]:
    """Two-player Shapley allocation of the joint behavior gain."""

    phi_m = 0.5 * ((effects.y10 - effects.y00) + (effects.y11 - effects.y01))
    phi_n = 0.5 * ((effects.y01 - effects.y00) + (effects.y11 - effects.y10))
    return {"m": float(phi_m), "n": float(phi_n), "total": float(phi_m + phi_n)}
