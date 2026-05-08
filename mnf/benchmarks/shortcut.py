from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import numpy as np


@dataclass
class ShortcutDataset:
    activations: np.ndarray
    labels: np.ndarray
    environments: np.ndarray
    causal_values: np.ndarray
    shortcut_values: np.ndarray
    causal_direction: np.ndarray
    shortcut_direction: np.ndarray

    def mask(self, environment: str) -> np.ndarray:
        return self.environments == environment


@dataclass(frozen=True)
class CandidateInvarianceScore:
    name: str
    train_accuracy: float
    shifted_accuracy: float
    invariance_gap: float
    min_environment_accuracy: float
    accepted: bool


def _orthonormal_pair(rng: np.random.Generator, dim: int) -> tuple[np.ndarray, np.ndarray]:
    a = rng.normal(size=dim)
    a /= np.linalg.norm(a)
    b = rng.normal(size=dim)
    b -= a * float(a @ b)
    b /= np.linalg.norm(b)
    return a, b


def _signed_binary(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.choice(np.array([-1.0, 1.0]), size=n)


def _shortcut_from_label(
    rng: np.random.Generator,
    y_signed: np.ndarray,
    agreement: float,
) -> np.ndarray:
    flip = rng.random(len(y_signed)) > agreement
    shortcut = y_signed.copy()
    shortcut[flip] *= -1.0
    return shortcut


def make_shortcut_dataset(
    n_per_environment: int = 1000,
    activation_dim: int = 12,
    causal_strength: float = 1.0,
    shortcut_strength: float = 1.0,
    train_shortcut_agreement: float = 0.98,
    shifted_shortcut_agreement: float = 0.1,
    noise: float = 0.1,
    seed: int = 0,
) -> ShortcutDataset:
    """Build a spurious-correlation control with an environment shift.

    The true label is the causal variable in both environments.  The shortcut
    is nearly equivalent to the label in the training environment, then mostly
    anti-correlated in the shifted environment.  A labelability-only method can
    select the shortcut; an invariant mechanism should select the causal state.
    """

    rng = np.random.default_rng(seed)
    causal_dir, shortcut_dir = _orthonormal_pair(rng, activation_dim)
    y_train = _signed_binary(rng, n_per_environment)
    y_shifted = _signed_binary(rng, n_per_environment)
    shortcut_train = _shortcut_from_label(rng, y_train, train_shortcut_agreement)
    shortcut_shifted = _shortcut_from_label(rng, y_shifted, shifted_shortcut_agreement)
    causal_values = np.concatenate([y_train, y_shifted])
    shortcut_values = np.concatenate([shortcut_train, shortcut_shifted])
    labels = (causal_values > 0).astype(int)
    environments = np.array(["train"] * n_per_environment + ["shifted"] * n_per_environment)
    activations = (
        causal_strength * causal_values[:, None] * causal_dir[None, :]
        + shortcut_strength * shortcut_values[:, None] * shortcut_dir[None, :]
        + noise * rng.normal(size=(2 * n_per_environment, activation_dim))
    )
    return ShortcutDataset(
        activations=activations,
        labels=labels,
        environments=environments,
        causal_values=causal_values,
        shortcut_values=shortcut_values,
        causal_direction=causal_dir,
        shortcut_direction=shortcut_dir,
    )


def _oriented_accuracy(values: np.ndarray, labels: np.ndarray, orientation: float) -> float:
    pred = ((orientation * values) > 0.0).astype(int)
    return float(np.mean(pred == labels))


def _fit_orientation(values: np.ndarray, labels: np.ndarray) -> float:
    acc_pos = _oriented_accuracy(values, labels, 1.0)
    acc_neg = _oriented_accuracy(values, labels, -1.0)
    return 1.0 if acc_pos >= acc_neg else -1.0


def score_shortcut_candidates(
    dataset: ShortcutDataset,
    min_accuracy: float = 0.8,
    max_gap: float = 0.15,
) -> dict[str, CandidateInvarianceScore]:
    """Score causal vs shortcut candidates by train fit and environment stability."""

    projections: Mapping[str, np.ndarray] = {
        "causal": dataset.activations @ dataset.causal_direction,
        "shortcut": dataset.activations @ dataset.shortcut_direction,
    }
    train_mask = dataset.mask("train")
    shifted_mask = dataset.mask("shifted")
    out: dict[str, CandidateInvarianceScore] = {}
    for name, values in projections.items():
        orientation = _fit_orientation(values[train_mask], dataset.labels[train_mask])
        train_acc = _oriented_accuracy(values[train_mask], dataset.labels[train_mask], orientation)
        shifted_acc = _oriented_accuracy(values[shifted_mask], dataset.labels[shifted_mask], orientation)
        gap = abs(train_acc - shifted_acc)
        min_acc = min(train_acc, shifted_acc)
        out[name] = CandidateInvarianceScore(
            name=name,
            train_accuracy=train_acc,
            shifted_accuracy=shifted_acc,
            invariance_gap=gap,
            min_environment_accuracy=min_acc,
            accepted=min_acc >= min_accuracy and gap <= max_gap,
        )
    return out


def shortcut_control(seed: int = 0) -> dict[str, float]:
    dataset = make_shortcut_dataset(seed=seed)
    scores = score_shortcut_candidates(dataset)
    causal = scores["causal"]
    shortcut = scores["shortcut"]
    return {
        "causal_train_accuracy": causal.train_accuracy,
        "causal_shifted_accuracy": causal.shifted_accuracy,
        "causal_invariance_gap": causal.invariance_gap,
        "shortcut_train_accuracy": shortcut.train_accuracy,
        "shortcut_shifted_accuracy": shortcut.shifted_accuracy,
        "shortcut_invariance_gap": shortcut.invariance_gap,
        "causal_accepted": float(causal.accepted),
        "shortcut_accepted": float(shortcut.accepted),
    }
