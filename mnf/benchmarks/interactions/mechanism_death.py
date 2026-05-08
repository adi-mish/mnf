from __future__ import annotations

import numpy as np

from mnf.interactions.dynamics import CoupledDynamicsParams, simulate_coupled_dynamics


def mechanism_death_metrics(seed: int = 0) -> dict[str, float]:
    params = CoupledDynamicsParams(
        intrinsic=np.array([1.1, 0.5]),
        support=np.zeros((2, 2)),
        competition=np.array([[0.0, 0.0], [1.4, 0.0]]),
        capacity_cost=np.zeros(2),
        dt=0.12,
    )
    trace = simulate_coupled_dynamics(np.array([0.08, 0.55]), params, n_steps=180, seed=seed)
    return {
        "winner_final_strength": float(trace.strengths[-1, 0]),
        "suppressed_final_strength": float(trace.strengths[-1, 1]),
        "suppressed_start_strength": float(trace.strengths[0, 1]),
        "suppression_delta": float(trace.strengths[0, 1] - trace.strengths[-1, 1]),
    }
