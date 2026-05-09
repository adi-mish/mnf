from __future__ import annotations

from collections.abc import Mapping

from mnf.interactions import (
    FactorialEffects,
    StructuralInteractionEvidence,
    factorial_from_callable,
    structural_interaction_evidence,
)


def directed_gate_behavior(gate: bool, worker: bool) -> float:
    return float(gate and worker)


def symmetric_synergy_behavior(left: bool, right: bool) -> float:
    return float(left and right)


def saturation_behavior(left: bool, right: bool) -> float:
    return min(1.0, float(left) + float(right))


def directed_gate_internal_trace(state: Mapping[str, bool]) -> dict[str, float]:
    gate = float(state["gate"])
    worker = float(state["worker"])
    return {
        "gate_signal": gate,
        "worker_signal": worker,
        "gated_worker": gate * worker,
        "output": gate * worker,
    }


def symmetric_internal_trace(state: Mapping[str, bool]) -> dict[str, float]:
    left = float(state["left"])
    right = float(state["right"])
    return {
        "left_signal": left,
        "right_signal": right,
        "joint_signal": left * right,
        "output": left * right,
    }


def directed_gate_internal_evidence() -> tuple[float, float]:
    """Return orientation evidence `(gate_to_worker, worker_to_gate)`.

    The downstream internal variable is `gated_worker`, which is naturally
    downstream of the worker pathway. The gate modulates worker's effect on that
    variable; worker does not modulate the pre-gate `worker_signal`.
    """

    def gated_worker(gate: bool, worker: bool) -> float:
        return directed_gate_internal_trace({"gate": gate, "worker": worker})["gated_worker"]

    def worker_signal(gate: bool, worker: bool) -> float:
        return directed_gate_internal_trace({"gate": gate, "worker": worker})["worker_signal"]

    gate_to_worker = FactorialEffects(
        y00=gated_worker(False, False),
        y10=gated_worker(True, False),
        y01=gated_worker(False, True),
        y11=gated_worker(True, True),
    ).gate_m_to_n
    worker_to_gate = FactorialEffects(
        y00=worker_signal(False, False),
        y10=worker_signal(True, False),
        y01=worker_signal(False, True),
        y11=worker_signal(True, True),
    ).gate_n_to_m
    return float(gate_to_worker), float(worker_to_gate)


def symmetric_internal_evidence() -> tuple[float, float]:
    """Symmetric AND has no privileged internal downstream direction."""

    return 0.0, 0.0


def table_aliasing_metrics() -> dict[str, object]:
    directed = factorial_from_callable(directed_gate_behavior)
    symmetric = factorial_from_callable(symmetric_synergy_behavior)
    saturated = factorial_from_callable(saturation_behavior)
    behavior_only = structural_interaction_evidence(directed)
    gate_to_worker, worker_to_gate = directed_gate_internal_evidence()
    directed_structural = structural_interaction_evidence(
        directed,
        internal_gate_m_to_n=gate_to_worker,
        internal_gate_n_to_m=worker_to_gate,
    )
    symmetric_structural = structural_interaction_evidence(
        symmetric,
        internal_gate_m_to_n=symmetric_internal_evidence()[0],
        internal_gate_n_to_m=symmetric_internal_evidence()[1],
    )
    return {
        "directed_table": directed.as_dict(),
        "symmetric_table": symmetric.as_dict(),
        "saturation_table": saturated.as_dict(),
        "directed_equals_symmetric_table": directed == symmetric,
        "behavior_only": behavior_only.as_dict(),
        "directed_structural": directed_structural.as_dict(),
        "symmetric_structural": symmetric_structural.as_dict(),
        "gate_to_worker_internal": float(gate_to_worker),
        "worker_to_gate_internal": float(worker_to_gate),
    }


def structural_aliasing_report() -> dict[str, StructuralInteractionEvidence]:
    directed = factorial_from_callable(directed_gate_behavior)
    gate_to_worker, worker_to_gate = directed_gate_internal_evidence()
    return {
        "behavior_only": structural_interaction_evidence(directed),
        "with_internal_evidence": structural_interaction_evidence(
            directed,
            internal_gate_m_to_n=gate_to_worker,
            internal_gate_n_to_m=worker_to_gate,
        ),
    }
