from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from mnf.benchmarks.cyclic import make_weekday_rotation_dataset, weekday_space
from mnf.charts.cyclic import CyclicChart
from mnf.core.types import CyclicSpace


@dataclass
class ScalarCyclicRegressor:
    """A deliberately naive scalar chart for cyclic variables.

    It maps activations to integer labels with least squares. This can be
    decodable on line-like variables, but it is not a natural chart for a circle:
    the wraparound between `period - 1` and `0` creates a discontinuity.
    """

    space: CyclicSpace
    weights: np.ndarray
    bias: float
    mean: np.ndarray

    @classmethod
    def fit(
        cls,
        activations: np.ndarray,
        labels: Sequence[Any],
        space: CyclicSpace,
    ) -> "ScalarCyclicRegressor":
        x = np.asarray(activations, dtype=float)
        if len(labels) != len(x):
            raise ValueError("labels and activations length mismatch")
        mean = x.mean(axis=0)
        xc = x - mean
        y = np.array([space.index(label) for label in labels], dtype=float)
        X = np.concatenate([xc, np.ones((len(xc), 1))], axis=1)
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        return cls(space=space, weights=beta[:-1], bias=float(beta[-1]), mean=mean)

    def scalar(self, activations: np.ndarray) -> np.ndarray:
        x = np.asarray(activations, dtype=float)
        return (x - self.mean) @ self.weights + self.bias

    def predict_index(self, activations: np.ndarray) -> np.ndarray:
        return np.rint(self.scalar(activations)).astype(int) % self.space.period

    def label_error(self, activations: np.ndarray, labels: Sequence[Any]) -> float:
        pred = self.predict_index(activations)
        expected = np.array([self.space.index(label) for label in labels], dtype=int)
        return float(np.mean(pred != expected))

    def scalar_shift_error(self, activations: np.ndarray, labels: Sequence[Any], steps: int) -> float:
        pred = (np.rint(self.scalar(activations) + steps).astype(int) % self.space.period)
        expected = np.array(
            [(self.space.index(label) + steps) % self.space.period for label in labels],
            dtype=int,
        )
        return float(np.mean(pred != expected))

    def description_length(self) -> float:
        return float(self.weights.size + 1) / 64.0


def compare_cyclic_baselines(
    n: int = 1200,
    noise: float = 0.03,
    seed: int = 0,
    train_fraction: float = 0.7,
) -> dict[str, float]:
    """Compare a typed cyclic chart against a scalar cyclic baseline."""

    ds = make_weekday_rotation_dataset(n=n, noise=noise, seed=seed)
    split = max(1, min(n - 1, int(train_fraction * n)))
    x_train, x_test = ds.activations[:split], ds.activations[split:]
    y_train, y_test = ds.labels[:split], ds.labels[split:]

    typed = CyclicChart.fit(x_train, y_train, weekday_space)
    typed_pred = typed.encode_label(x_test)
    typed_label_error = sum(a != b for a, b in zip(typed_pred, y_test)) / len(y_test)
    typed_rotation_error = typed.intervention_error_after_rotation(x_test, y_test, ds.rotation_steps)

    scalar = ScalarCyclicRegressor.fit(x_train, y_train, weekday_space)
    scalar_label_error = scalar.label_error(x_test, y_test)
    scalar_rotation_error = scalar.scalar_shift_error(x_test, y_test, ds.rotation_steps)

    return {
        "typed_label_error": float(typed_label_error),
        "typed_rotation_error": float(typed_rotation_error),
        "typed_description_length": typed.encoder_matrix.size / 64.0,
        "scalar_label_error": scalar_label_error,
        "scalar_rotation_error": scalar_rotation_error,
        "scalar_description_length": scalar.description_length(),
    }
