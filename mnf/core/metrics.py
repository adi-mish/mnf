from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence
import math
import numpy as np


def mse(a: Any, b: Any) -> float:
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    return float(np.mean((arr_a - arr_b) ** 2))


def normalized_mse(a: Any, b: Any, eps: float = 1e-12) -> float:
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    denom = float(np.mean(arr_a**2) + eps)
    return float(np.mean((arr_a - arr_b) ** 2) / denom)


def classification_error(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")
    if not y_true:
        return 0.0
    return sum(a != b for a, b in zip(y_true, y_pred)) / len(y_true)


@dataclass(frozen=True)
class GraphF1:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def graph_f1(true_edges: Iterable[tuple[str, str]], pred_edges: Iterable[tuple[str, str]]) -> GraphF1:
    true_set = set(true_edges)
    pred_set = set(pred_edges)
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    precision = tp / (tp + fp) if tp + fp > 0 else 1.0 if not true_set else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 1.0 if not pred_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return GraphF1(precision, recall, f1, tp, fp, fn)


def intervention_prediction_error(
    observed: Sequence[Mapping[str, Any]],
    predicted: Sequence[Mapping[str, Any]],
    keys: Sequence[str] | None = None,
) -> float:
    if len(observed) != len(predicted):
        raise ValueError("observed and predicted must have same length")
    if len(observed) == 0:
        return 0.0
    if keys is None:
        keys = tuple(observed[0].keys())
    errs = []
    for obs, pred in zip(observed, predicted):
        for k in keys:
            a, b = obs[k], pred[k]
            if isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number)):
                errs.append((float(a) - float(b)) ** 2)
            else:
                errs.append(0.0 if a == b else 1.0)
    return float(np.mean(errs))


def description_length_from_counts(
    n_nodes: int,
    n_edges: int,
    n_state_dims: int,
    n_params: int = 0,
    node_cost: float = 1.0,
    edge_cost: float = 0.5,
    dim_cost: float = 0.1,
    param_cost: float = 0.01,
) -> float:
    return float(node_cost * n_nodes + edge_cost * n_edges + dim_cost * n_state_dims + param_cost * n_params)


def correlation_matrix(x: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = x if y is None else np.asarray(y, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    xz = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-12)
    yz = (y - y.mean(axis=0)) / (y.std(axis=0) + 1e-12)
    return xz.T @ yz / max(1, x.shape[0] - 1)


def topk_indices(values: np.ndarray, k: int) -> np.ndarray:
    values = np.asarray(values)
    if k <= 0:
        return np.array([], dtype=int)
    k = min(k, values.size)
    idx = np.argpartition(-values, k - 1)[:k]
    return idx[np.argsort(-values[idx])]
