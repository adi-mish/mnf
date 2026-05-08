from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FactorialEffects:
    """Pairwise on/off intervention table for two candidate mechanisms.

    The convention is `y_ab`, where `a` indicates whether mechanism `m` is on
    and `b` indicates whether mechanism `n` is on. Larger values are assumed to
    mean stronger or better behavior.
    """

    y00: float
    y10: float
    y01: float
    y11: float

    @property
    def joint_effect(self) -> float:
        return self.y11 - self.y00

    @property
    def effect_m_when_n_off(self) -> float:
        return self.y10 - self.y00

    @property
    def effect_m_when_n_on(self) -> float:
        return self.y11 - self.y01

    @property
    def effect_n_when_m_off(self) -> float:
        return self.y01 - self.y00

    @property
    def effect_n_when_m_on(self) -> float:
        return self.y11 - self.y10

    @property
    def drop_m_from_joint(self) -> float:
        return self.y11 - self.y01

    @property
    def drop_n_from_joint(self) -> float:
        return self.y11 - self.y10

    @property
    def dual_ablation_drop(self) -> float:
        return self.y11 - self.y00

    @property
    def synergy(self) -> float:
        """Additive interaction term.

        Positive values indicate super-additive cooperation. Negative values
        indicate sub-additivity, which includes redundancy and competition.
        """

        return self.y11 - self.y10 - self.y01 + self.y00

    @property
    def compensation_score(self) -> float:
        """How much dual ablation matters after single ablations do not.

        This is high for redundant paths: `y11 ~= y10 ~= y01 >> y00`.
        """

        single_drop_sum = max(0.0, self.drop_m_from_joint) + max(0.0, self.drop_n_from_joint)
        return max(0.0, self.dual_ablation_drop - single_drop_sum)

    @property
    def redundancy_score(self) -> float:
        """Score for substitutable mechanisms with little joint improvement."""

        single_floor = min(self.effect_m_when_n_off, self.effect_n_when_m_off)
        extra_joint_gain = max(0.0, self.y11 - max(self.y10, self.y01))
        return max(0.0, single_floor - extra_joint_gain)

    @property
    def competition_score(self) -> float:
        """Sub-additive interaction not already explained by compensation."""

        return max(0.0, -self.synergy - self.compensation_score)

    @property
    def gate_m_to_n(self) -> float:
        return self.effect_n_when_m_on - self.effect_n_when_m_off

    @property
    def gate_n_to_m(self) -> float:
        return self.effect_m_when_n_on - self.effect_m_when_n_off

    def as_dict(self) -> dict[str, float]:
        out = asdict(self)
        out.update(
            {
                "joint_effect": self.joint_effect,
                "effect_m_when_n_off": self.effect_m_when_n_off,
                "effect_m_when_n_on": self.effect_m_when_n_on,
                "effect_n_when_m_off": self.effect_n_when_m_off,
                "effect_n_when_m_on": self.effect_n_when_m_on,
                "drop_m_from_joint": self.drop_m_from_joint,
                "drop_n_from_joint": self.drop_n_from_joint,
                "dual_ablation_drop": self.dual_ablation_drop,
                "synergy": self.synergy,
                "compensation_score": self.compensation_score,
                "redundancy_score": self.redundancy_score,
                "competition_score": self.competition_score,
                "gate_m_to_n": self.gate_m_to_n,
                "gate_n_to_m": self.gate_n_to_m,
            }
        )
        return out


def factorial_from_callable(fn: Callable[[bool, bool], float]) -> FactorialEffects:
    return FactorialEffects(
        y00=float(fn(False, False)),
        y10=float(fn(True, False)),
        y01=float(fn(False, True)),
        y11=float(fn(True, True)),
    )


def classify_pairwise_interaction(effects: FactorialEffects, tol: float = 1e-6) -> str:
    """Return a coarse interaction label for a pairwise intervention table."""

    if effects.compensation_score > tol and effects.redundancy_score > tol:
        return "redundant"
    if effects.synergy > tol:
        return "synergistic_or_gated"
    if effects.competition_score > tol:
        return "competitive"
    if abs(effects.joint_effect) <= tol:
        return "inactive"
    return "additive"
