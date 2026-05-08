from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from mnf.interactions.factorial_effects import FactorialEffects


@dataclass(frozen=True)
class EffectInterval:
    mean: float
    low: float
    high: float

    @property
    def width(self) -> float:
        return self.high - self.low

    def as_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "low": self.low, "high": self.high, "width": self.width}


def bootstrap_factorial_effects(
    samples_by_cell: Mapping[str, Sequence[float]],
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, EffectInterval]:
    """Bootstrap uncertainty for derived pairwise interaction quantities."""

    required = ("y00", "y10", "y01", "y11")
    arrays = {key: np.asarray(samples_by_cell[key], dtype=float) for key in required}
    if any(arr.size == 0 for arr in arrays.values()):
        raise ValueError("all factorial cells must contain at least one sample")

    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {
        "joint_effect": [],
        "synergy": [],
        "redundancy_score": [],
        "compensation_score": [],
        "gate_m_to_n": [],
        "gate_n_to_m": [],
    }
    for _ in range(n_boot):
        means = {
            key: float(np.mean(arr[rng.integers(0, arr.size, size=arr.size)]))
            for key, arr in arrays.items()
        }
        effects = FactorialEffects(**means)
        draws["joint_effect"].append(effects.joint_effect)
        draws["synergy"].append(effects.synergy)
        draws["redundancy_score"].append(effects.redundancy_score)
        draws["compensation_score"].append(effects.compensation_score)
        draws["gate_m_to_n"].append(effects.gate_m_to_n)
        draws["gate_n_to_m"].append(effects.gate_n_to_m)

    alpha = (1.0 - confidence) / 2.0
    out: dict[str, EffectInterval] = {}
    for key, values in draws.items():
        arr = np.asarray(values, dtype=float)
        out[key] = EffectInterval(
            mean=float(np.mean(arr)),
            low=float(np.quantile(arr, alpha)),
            high=float(np.quantile(arr, 1.0 - alpha)),
        )
    return out


def noisy_factorial_samples(
    effects: FactorialEffects,
    n: int = 256,
    noise: float = 0.03,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "y00": effects.y00 + noise * rng.normal(size=n),
        "y10": effects.y10 + noise * rng.normal(size=n),
        "y01": effects.y01 + noise * rng.normal(size=n),
        "y11": effects.y11 + noise * rng.normal(size=n),
    }
