from __future__ import annotations

from mnf.interactions.factorial_effects import classify_pairwise_interaction, factorial_from_callable


def gated_worker_behavior(gate_on: bool, worker_on: bool, gate_strength: float = 1.0) -> float:
    return float(gate_strength if gate_on and worker_on else 0.0)


def gating_metrics(gate_strength: float = 1.0) -> dict[str, float | str]:
    effects = factorial_from_callable(lambda gate, worker: gated_worker_behavior(gate, worker, gate_strength))
    out: dict[str, float | str] = effects.as_dict()
    out["interaction_label"] = classify_pairwise_interaction(effects)
    out["worker_effect_when_gate_on"] = effects.effect_n_when_m_on
    out["worker_effect_when_gate_off"] = effects.effect_n_when_m_off
    return out
