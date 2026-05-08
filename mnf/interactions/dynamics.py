from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CoupledDynamicsParams:
    intrinsic: np.ndarray
    support: np.ndarray
    competition: np.ndarray
    capacity_cost: np.ndarray
    dt: float = 0.1
    noise: float = 0.0


@dataclass(frozen=True)
class CoupledDynamicsTrace:
    steps: np.ndarray
    strengths: np.ndarray
    utility: np.ndarray


def simulate_coupled_dynamics(
    initial_strengths: np.ndarray,
    params: CoupledDynamicsParams,
    n_steps: int = 160,
    seed: int = 0,
) -> CoupledDynamicsTrace:
    """Simulate a simple mechanism ecology equation on [0, 1]^K."""

    rng = np.random.default_rng(seed)
    s = np.asarray(initial_strengths, dtype=float).copy()
    intrinsic = np.asarray(params.intrinsic, dtype=float)
    support = np.asarray(params.support, dtype=float)
    competition = np.asarray(params.competition, dtype=float)
    capacity_cost = np.asarray(params.capacity_cost, dtype=float)
    k = s.size
    if intrinsic.shape != (k,) or capacity_cost.shape != (k,):
        raise ValueError("intrinsic and capacity_cost must match initial strengths")
    if support.shape != (k, k) or competition.shape != (k, k):
        raise ValueError("support and competition must be K x K matrices")

    strengths = np.zeros((n_steps, k), dtype=float)
    utilities = np.zeros((n_steps, k), dtype=float)
    for t in range(n_steps):
        strengths[t] = s
        utility = intrinsic + support @ s - competition @ s - capacity_cost
        utilities[t] = utility
        ds = s * (1.0 - s) * utility
        if params.noise > 0:
            ds = ds + params.noise * rng.normal(size=k)
        s = np.clip(s + params.dt * ds, 0.0, 1.0)
    return CoupledDynamicsTrace(steps=np.arange(n_steps, dtype=float), strengths=strengths, utility=utilities)


def first_crossing_step(trace: CoupledDynamicsTrace, mechanism_index: int, threshold: float = 0.5) -> float:
    hits = np.flatnonzero(trace.strengths[:, mechanism_index] >= threshold)
    return float(trace.steps[hits[0]]) if len(hits) else float("nan")
