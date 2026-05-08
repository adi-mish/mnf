from __future__ import annotations


def support_delta(mechanisticity_intact: float, mechanisticity_after_support_ablation: float) -> float:
    """Positive when one mechanism supports another mechanism's mechanisticity."""

    return float(mechanisticity_intact - mechanisticity_after_support_ablation)


def normalized_support_delta(
    mechanisticity_intact: float,
    mechanisticity_after_support_ablation: float,
    eps: float = 1e-12,
) -> float:
    return support_delta(mechanisticity_intact, mechanisticity_after_support_ablation) / (
        abs(float(mechanisticity_intact)) + eps
    )
