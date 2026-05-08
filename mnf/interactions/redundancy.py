from __future__ import annotations

from mnf.interactions.factorial_effects import FactorialEffects


def redundancy_from_factorial(effects: FactorialEffects) -> float:
    return effects.redundancy_score


def compensation_from_factorial(effects: FactorialEffects) -> float:
    return effects.compensation_score


def single_ablation_misses_pair(effects: FactorialEffects, single_tol: float = 1e-6, dual_threshold: float = 0.1) -> bool:
    single_missed = abs(effects.drop_m_from_joint) <= single_tol and abs(effects.drop_n_from_joint) <= single_tol
    return bool(single_missed and effects.dual_ablation_drop >= dual_threshold)
