from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence
import math
import numpy as np

from mnf.core.types import CyclicSpace


@dataclass
class CyclicChart:
    """A typed chart for cyclic variables such as weekday/month/mod-n state.

    Fit is supervised in this scaffold: labels identify the cyclic variable, and
    the chart learns the minimum-norm linear map from activations to the 2D circle
    coordinates.  This is useful for benchmarking scalar-vs-typed explanations.
    """

    space: CyclicSpace
    encoder_matrix: np.ndarray  # activation_dim x 2
    decoder_matrix: np.ndarray  # 2 x activation_dim
    mean: np.ndarray
    radius: float = 1.0
    name: str = "cyclic_chart"

    @classmethod
    def fit(cls, activations: np.ndarray, labels: Sequence[Any], space: CyclicSpace) -> "CyclicChart":
        x = np.asarray(activations, dtype=float)
        if len(labels) != len(x):
            raise ValueError("labels and activations length mismatch")
        mean = x.mean(axis=0)
        xc = x - mean
        y = np.stack([space.embed2(v) for v in labels], axis=0)
        # Ridge-stabilized least squares from activation to circle and back.
        ridge = 1e-6
        encoder = np.linalg.solve(xc.T @ xc + ridge * np.eye(xc.shape[1]), xc.T @ y)
        decoder = np.linalg.solve(y.T @ y + ridge * np.eye(2), y.T @ xc)
        radius = float(np.mean(np.linalg.norm(y, axis=1)))
        return cls(space=space, encoder_matrix=encoder, decoder_matrix=decoder, mean=mean, radius=radius)

    def encode_xy(self, activations: np.ndarray) -> np.ndarray:
        x = np.asarray(activations, dtype=float)
        return (x - self.mean) @ self.encoder_matrix

    def encode_angle(self, activations: np.ndarray) -> np.ndarray:
        xy = self.encode_xy(activations)
        return np.arctan2(xy[:, 1], xy[:, 0]) % (2.0 * math.pi)

    def encode_label(self, activations: np.ndarray) -> list[Any]:
        return [self.space.nearest_from_angle(theta) for theta in self.encode_angle(activations)]

    def decode_xy(self, xy: np.ndarray) -> np.ndarray:
        return np.asarray(xy, dtype=float) @ self.decoder_matrix + self.mean

    def decode_label(self, labels: Sequence[Any]) -> np.ndarray:
        xy = np.stack([self.space.embed2(v, radius=self.radius) for v in labels], axis=0)
        return self.decode_xy(xy)

    def rotate(self, activations: np.ndarray, steps: int) -> np.ndarray:
        xy = self.encode_xy(activations)
        angle = 2.0 * math.pi * steps / self.space.period
        rot = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        xy2 = xy @ rot.T
        return self.decode_xy(xy2)

    def intervention_error_after_rotation(self, activations: np.ndarray, labels: Sequence[Any], steps: int) -> float:
        rotated = self.rotate(activations, steps)
        pred = self.encode_label(rotated)
        expected = [self.space.project(self.space.index(v) + steps) for v in labels]
        return sum(p != e for p, e in zip(pred, expected)) / max(1, len(expected))
