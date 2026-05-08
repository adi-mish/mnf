from __future__ import annotations

from collections.abc import Mapping, Sequence


def contrast_error_bound(coefficients: Sequence[float], cell_error_bound: float) -> float:
    """Worst-case absolute error for a linear contrast of cell means."""

    if cell_error_bound < 0:
        raise ValueError("cell_error_bound must be nonnegative")
    return float(sum(abs(float(c)) for c in coefficients) * cell_error_bound)


def pairwise_contrast_error_bounds(cell_error_bound: float) -> dict[str, float]:
    """Worst-case bounds for the linear pairwise factorial contrasts."""

    return {
        "joint_effect": contrast_error_bound((-1.0, 0.0, 0.0, 1.0), cell_error_bound),
        "effect_m_when_n_off": contrast_error_bound((-1.0, 1.0, 0.0, 0.0), cell_error_bound),
        "effect_m_when_n_on": contrast_error_bound((0.0, 0.0, -1.0, 1.0), cell_error_bound),
        "effect_n_when_m_off": contrast_error_bound((-1.0, 0.0, 1.0, 0.0), cell_error_bound),
        "effect_n_when_m_on": contrast_error_bound((0.0, -1.0, 0.0, 1.0), cell_error_bound),
        "synergy": contrast_error_bound((1.0, -1.0, -1.0, 1.0), cell_error_bound),
        "gate_m_to_n": contrast_error_bound((1.0, -1.0, -1.0, 1.0), cell_error_bound),
        "gate_n_to_m": contrast_error_bound((1.0, -1.0, -1.0, 1.0), cell_error_bound),
    }


def higher_order_contrast_error_bound(order: int, cell_error_bound: float) -> float:
    if order <= 0:
        raise ValueError("order must be positive")
    return float((2**order) * cell_error_bound)


def sign_is_stable(value: float, error_bound: float, margin: float = 0.0) -> bool:
    """Whether a contrast's sign is stable under the given error bound."""

    return bool(abs(float(value)) > float(error_bound) + float(margin))


def all_required_contrasts_stable(
    contrasts: Mapping[str, float],
    bounds: Mapping[str, float],
    required: Sequence[str],
    margin: float = 0.0,
) -> bool:
    return all(sign_is_stable(float(contrasts[name]), float(bounds[name]), margin=margin) for name in required)
