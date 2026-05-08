"""Lightweight baseline selectors for MechanismLab experiments."""

from mnf.baselines.selection import (
    SelectorResult,
    compare_shortcut_selectors,
    select_by_invariance,
    select_by_train_accuracy,
)
from mnf.baselines.cyclic import ScalarCyclicRegressor, compare_cyclic_baselines

__all__ = [
    "ScalarCyclicRegressor",
    "SelectorResult",
    "compare_cyclic_baselines",
    "compare_shortcut_selectors",
    "select_by_invariance",
    "select_by_train_accuracy",
]
