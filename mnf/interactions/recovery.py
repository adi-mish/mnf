from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from mnf.core.metrics import GraphF1


@dataclass(frozen=True)
class InteractionRecoveryReport:
    precision: float
    recall: float
    f1: float
    accuracy: float
    tp: int
    fp: int
    fn: int
    correct_labels: int
    total_labels: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "correct_labels": self.correct_labels,
            "total_labels": self.total_labels,
        }


def threshold_edges(
    names: Sequence[str],
    matrix: np.ndarray,
    threshold: float = 1e-6,
    directed: bool = False,
) -> set[tuple[str, str]]:
    arr = np.asarray(matrix, dtype=float)
    if arr.shape != (len(names), len(names)):
        raise ValueError("matrix must be K x K")
    edges: set[tuple[str, str]] = set()
    for i, src in enumerate(names):
        for j, dst in enumerate(names):
            if i == j:
                continue
            if not directed and j <= i:
                continue
            if abs(float(arr[i, j])) > threshold:
                edges.add((src, dst))
    return edges


def edge_f1(true_edges: set[tuple[str, str]], pred_edges: set[tuple[str, str]]) -> GraphF1:
    tp = len(true_edges & pred_edges)
    fp = len(pred_edges - true_edges)
    fn = len(true_edges - pred_edges)
    precision = tp / (tp + fp) if tp + fp else 1.0 if not true_edges else 0.0
    recall = tp / (tp + fn) if tp + fn else 1.0 if not pred_edges else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return GraphF1(precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn)


def label_recovery_report(
    true_labels: Mapping[tuple[str, str], str],
    pred_labels: Mapping[tuple[str, str], str],
    inactive_label: str = "inactive",
) -> InteractionRecoveryReport:
    keys = set(true_labels) | set(pred_labels)
    true_active = {key for key in keys if true_labels.get(key, inactive_label) != inactive_label}
    pred_active = {key for key in keys if pred_labels.get(key, inactive_label) != inactive_label}
    graph = edge_f1(true_active, pred_active)
    correct = sum(true_labels.get(key, inactive_label) == pred_labels.get(key, inactive_label) for key in keys)
    total = len(keys)
    accuracy = correct / total if total else 1.0
    return InteractionRecoveryReport(
        precision=graph.precision,
        recall=graph.recall,
        f1=graph.f1,
        accuracy=float(accuracy),
        tp=graph.tp,
        fp=graph.fp,
        fn=graph.fn,
        correct_labels=int(correct),
        total_labels=int(total),
    )
