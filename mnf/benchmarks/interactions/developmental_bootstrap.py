from __future__ import annotations

import numpy as np

from mnf.interactions.dynamics import CoupledDynamicsParams, first_crossing_step, simulate_coupled_dynamics


def developmental_bootstrap_metrics(seed: int = 0) -> dict[str, float]:
    params = CoupledDynamicsParams(
        intrinsic=np.array([1.25, -0.35]),
        support=np.array([[0.0, 0.0], [1.45, 0.0]]),
        competition=np.zeros((2, 2)),
        capacity_cost=np.zeros(2),
        dt=0.12,
    )
    trace = simulate_coupled_dynamics(np.array([0.05, 0.02]), params, n_steps=220, seed=seed)
    scaffold_step = first_crossing_step(trace, 0, threshold=0.5)
    dependent_step = first_crossing_step(trace, 1, threshold=0.5)
    support_threshold = (0.35 / 1.45)
    support_hits = np.flatnonzero(trace.strengths[:, 0] > support_threshold)
    support_ready_step = float(trace.steps[support_hits[0]]) if len(support_hits) else float("nan")
    return {
        "scaffold_crossing_step": scaffold_step,
        "dependent_crossing_step": dependent_step,
        "support_ready_step": support_ready_step,
        "developmental_lag_steps": float(dependent_step - scaffold_step),
        "final_scaffold_strength": float(trace.strengths[-1, 0]),
        "final_dependent_strength": float(trace.strengths[-1, 1]),
    }
