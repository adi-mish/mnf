from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class TrainingDynamicsTrace:
    steps: np.ndarray
    mechanism_strength: np.ndarray
    intervention_score: np.ndarray
    behavioral_accuracy: np.ndarray
    behavioral_emerged: np.ndarray


def make_training_emergence_trace(
    n_steps: int = 120,
    mechanism_midpoint: float = 45.0,
    behavior_threshold: float = 0.68,
    mechanism_rate: float = 0.12,
    behavior_sharpness: float = 22.0,
    noise: float = 0.01,
    seed: int = 0,
) -> TrainingDynamicsTrace:
    """Synthetic training trace where mechanism forms before behavior jumps.

    This models a common emergence pattern: the internal mechanism improves
    gradually, while the measured behavior is thresholded and appears sudden.
    The intervention score is a noisy mechanistic progress proxy.
    """

    rng = np.random.default_rng(seed)
    steps = np.arange(n_steps, dtype=float)
    strength = 1.0 / (1.0 + np.exp(-mechanism_rate * (steps - mechanism_midpoint)))
    intervention = np.clip(strength + noise * rng.normal(size=n_steps), 0.0, 1.0)
    behavior = 1.0 / (1.0 + np.exp(-behavior_sharpness * (strength - behavior_threshold)))
    behavior = np.clip(behavior + noise * rng.normal(size=n_steps), 0.0, 1.0)
    emerged = behavior >= 0.8
    return TrainingDynamicsTrace(
        steps=steps,
        mechanism_strength=strength,
        intervention_score=intervention,
        behavioral_accuracy=behavior,
        behavioral_emerged=emerged,
    )


def emergence_lead_metrics(
    trace: TrainingDynamicsTrace,
    mechanism_threshold: float = 0.5,
    behavior_threshold: float = 0.8,
) -> dict[str, float]:
    """Measure whether intervention progress crosses threshold before behavior."""

    mech_hits = np.flatnonzero(trace.intervention_score >= mechanism_threshold)
    behavior_hits = np.flatnonzero(trace.behavioral_accuracy >= behavior_threshold)
    mechanism_step = float(trace.steps[mech_hits[0]]) if len(mech_hits) else float("nan")
    behavior_step = float(trace.steps[behavior_hits[0]]) if len(behavior_hits) else float("nan")
    lead = behavior_step - mechanism_step
    corr = np.corrcoef(trace.intervention_score, trace.behavioral_accuracy)[0, 1]
    if not np.isfinite(corr):
        corr = 0.0
    return {
        "mechanism_crossing_step": mechanism_step,
        "behavior_crossing_step": behavior_step,
        "mechanism_lead_steps": float(lead),
        "mechanism_behavior_correlation": float(corr),
        "final_intervention_score": float(trace.intervention_score[-1]),
        "final_behavioral_accuracy": float(trace.behavioral_accuracy[-1]),
    }


def training_emergence_control(seed: int = 0) -> dict[str, float]:
    trace = make_training_emergence_trace(seed=seed)
    return emergence_lead_metrics(trace)
