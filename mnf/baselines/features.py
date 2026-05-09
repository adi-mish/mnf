from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class FeatureBaselineResult:
    name: str
    label_accuracy: float
    causal_use_score: float
    description_length: float
    false_mechanism: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "label_accuracy": self.label_accuracy,
            "causal_use_score": self.causal_use_score,
            "description_length": self.description_length,
            "false_mechanism": self.false_mechanism,
        }


def make_labelability_control(
    n: int = 1024,
    dim: int = 32,
    label_strength: float = 2.0,
    causal_output: bool = False,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=n)
    semantic = rng.normal(size=dim)
    semantic /= np.linalg.norm(semantic)
    x = rng.normal(size=(n, dim)) + label_strength * (2 * labels - 1)[:, None] * semantic[None, :]
    if causal_output:
        output = 1.5 * (2 * labels - 1) + 0.2 * rng.normal(size=n)
    else:
        output = rng.normal(size=n)
    return x, labels, output


def linear_probe_direction(x: np.ndarray, labels: np.ndarray) -> np.ndarray:
    y = 2 * labels - 1
    X = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    direction = np.linalg.lstsq(X, y, rcond=None)[0][:-1]
    return _normalize(direction)


def pca_first_direction(x: np.ndarray, labels: np.ndarray | None = None) -> np.ndarray:
    del labels
    centered = x - np.mean(x, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return _normalize(vt[0])


def random_search_direction(
    x: np.ndarray,
    labels: np.ndarray,
    n_directions: int = 256,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    best_direction: np.ndarray | None = None
    best_accuracy = -1.0
    for _ in range(n_directions):
        direction = _normalize(rng.normal(size=x.shape[1]))
        accuracy = _label_accuracy(x, labels, direction)
        flipped = _label_accuracy(x, labels, -direction)
        if flipped > accuracy:
            direction = -direction
            accuracy = flipped
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_direction = direction
    if best_direction is None:
        raise ValueError("n_directions must be positive")
    return best_direction


def _normalize(direction: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        return np.zeros_like(direction)
    return direction / norm


def _label_accuracy(x: np.ndarray, labels: np.ndarray, direction: np.ndarray) -> float:
    scores = x @ direction
    pred = (scores > np.median(scores)).astype(int)
    accuracy = float(np.mean(pred == labels))
    return max(accuracy, 1.0 - accuracy)


def _causal_use_score(x: np.ndarray, output: np.ndarray, direction: np.ndarray) -> float:
    scores = x @ direction
    corr = np.corrcoef(scores, output)[0, 1]
    if not np.isfinite(corr):
        return 0.0
    return float(abs(corr))


def score_feature_baseline(
    name: str,
    direction_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x: np.ndarray,
    labels: np.ndarray,
    output: np.ndarray,
    description_length: float,
) -> FeatureBaselineResult:
    direction = direction_fn(x, labels)
    label_accuracy = _label_accuracy(x, labels, direction)
    causal_use = _causal_use_score(x, output, direction)
    return FeatureBaselineResult(
        name=name,
        label_accuracy=label_accuracy,
        causal_use_score=causal_use,
        description_length=description_length,
        false_mechanism=bool(label_accuracy >= 0.9 and causal_use < 0.2),
    )


def compare_feature_baselines(seed: int = 0) -> dict[str, object]:
    baselines: tuple[tuple[str, Callable[[np.ndarray, np.ndarray], np.ndarray], float], ...] = (
        ("linear_probe", linear_probe_direction, 1.0),
        ("pca_first", pca_first_direction, 0.5),
        ("random_search", lambda x, y: random_search_direction(x, y, seed=seed), 2.0),
    )
    out: dict[str, object] = {}
    for setting, causal_output in (("random_labelable", False), ("trained_used", True)):
        x, labels, output = make_labelability_control(causal_output=causal_output, seed=seed)
        results = [
            score_feature_baseline(name, fn, x, labels, output, description_length=dl).as_dict()
            for name, fn, dl in baselines
        ]
        out[setting] = {
            "rows": results,
            "false_mechanism_rate": float(np.mean([row["false_mechanism"] for row in results])),
            "mean_label_accuracy": float(np.mean([row["label_accuracy"] for row in results])),
            "mean_causal_use_score": float(np.mean([row["causal_use_score"] for row in results])),
        }
    return out
