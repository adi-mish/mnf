from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from mnf.charts.transitions import MixtureOfLinearTransforms, fit_linear_transition
from mnf.core.metrics import mse


@dataclass
class TransitionOnlyDataset:
    x: np.ndarray
    y: np.ndarray
    gate: np.ndarray
    matrices: tuple[np.ndarray, np.ndarray]
    biases: tuple[np.ndarray, np.ndarray]


def make_transition_only_dataset(
    n: int = 2000,
    input_dim: int = 6,
    output_dim: int = 4,
    gate_separation: float = 2.5,
    noise: float = 0.02,
    seed: int = 0,
) -> TransitionOnlyDataset:
    """Piecewise-linear transform where the mechanism is a gated transition.

    There is no privileged sparse state feature that explains the output well;
    a global linear map is deliberately misspecified, while a transition atom
    with a gate can capture the mechanism.
    """

    rng = np.random.default_rng(seed)
    gate = rng.random(n) > 0.5
    x = rng.normal(size=(n, input_dim))
    x[gate, 0] += gate_separation
    x[~gate, 0] -= gate_separation
    w0 = rng.normal(scale=0.8, size=(input_dim, output_dim))
    w1 = rng.normal(scale=0.8, size=(input_dim, output_dim))
    b0 = rng.normal(scale=0.2, size=output_dim)
    b1 = rng.normal(scale=0.2, size=output_dim)
    y = np.empty((n, output_dim), dtype=float)
    y[~gate] = x[~gate] @ w0 + b0
    y[gate] = x[gate] @ w1 + b1
    y += noise * rng.normal(size=y.shape)
    return TransitionOnlyDataset(x=x, y=y, gate=gate, matrices=(w0, w1), biases=(b0, b1))


def compare_global_linear_vs_molt(
    dataset: TransitionOnlyDataset,
    n_atoms: int = 2,
    seed: int = 0,
) -> dict[str, float]:
    global_atom = fit_linear_transition(dataset.x, dataset.y)
    global_pred = global_atom(dataset.x)
    molt = MixtureOfLinearTransforms.fit_with_kmeans_gates(dataset.x, dataset.y, n_atoms=n_atoms, seed=seed)
    molt_pred = molt.predict(dataset.x)
    return {
        "global_linear_mse": mse(dataset.y, global_pred),
        "molt_mse": mse(dataset.y, molt_pred),
        "molt_description_length": molt.description_length(),
    }
