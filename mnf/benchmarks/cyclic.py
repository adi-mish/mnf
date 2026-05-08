from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from mnf.core.types import CyclicSpace

WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
weekday_space = CyclicSpace(period=7, labels=WEEKDAYS, name="weekday")


@dataclass
class CyclicDataset:
    activations: np.ndarray
    labels: list[str]
    rotated_labels: list[str]
    rotation_steps: int
    true_plane: np.ndarray
    mean: np.ndarray


def make_weekday_rotation_dataset(
    n: int = 700,
    activation_dim: int = 16,
    rotation_steps: int = 2,
    noise: float = 0.03,
    seed: int = 0,
) -> CyclicDataset:
    """Generate activations with a true 2D circular weekday feature."""
    rng = np.random.default_rng(seed)
    labels = [WEEKDAYS[i % 7] for i in rng.integers(0, 7, size=n)]
    # Random orthonormal 2D plane in activation_dim.
    plane = rng.normal(size=(2, activation_dim))
    q, _ = np.linalg.qr(plane.T)
    plane = q[:, :2].T
    mean = rng.normal(scale=0.1, size=activation_dim)
    xy = np.stack([weekday_space.embed2(label) for label in labels], axis=0)
    activations = xy @ plane + mean + noise * rng.normal(size=(n, activation_dim))
    rotated = [weekday_space.project(weekday_space.index(label) + rotation_steps) for label in labels]
    return CyclicDataset(activations, labels, rotated, rotation_steps, plane, mean)
