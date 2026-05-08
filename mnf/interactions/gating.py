from __future__ import annotations

from mnf.interactions.factorial_effects import FactorialEffects


def conditional_gate_strength(effect_when_gate_on: float, effect_when_gate_off: float) -> float:
    """Difference in a mechanism's behavioral effect across gate states."""

    return float(effect_when_gate_on - effect_when_gate_off)


def gate_m_to_n_from_factorial(effects: FactorialEffects) -> float:
    return effects.gate_m_to_n


def gate_n_to_m_from_factorial(effects: FactorialEffects) -> float:
    return effects.gate_n_to_m
