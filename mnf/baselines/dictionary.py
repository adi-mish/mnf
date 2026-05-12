from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mnf.baselines.features import _causal_use_score, _label_accuracy, _normalize, make_labelability_control


@dataclass(frozen=True)
class DictionaryBaselineRow:
    name: str
    label_accuracy: float
    causal_use_score: float
    reconstruction_error: float
    false_mechanism: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "label_accuracy": self.label_accuracy,
            "causal_use_score": self.causal_use_score,
            "reconstruction_error": self.reconstruction_error,
            "false_mechanism": self.false_mechanism,
        }


def ica_first_direction(x: np.ndarray, labels: np.ndarray | None = None, seed: int = 0) -> np.ndarray:
    del labels
    try:
        from sklearn.decomposition import FastICA
    except Exception:
        centered = x - np.mean(x, axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        return _normalize(vt[0])
    model = FastICA(n_components=1, random_state=seed, whiten="unit-variance", max_iter=1000, tol=1e-4)
    model.fit(np.asarray(x, dtype=float))
    return _normalize(model.components_[0])


def kmeans_dictionary_direction(x: np.ndarray, labels: np.ndarray, n_clusters: int = 2, seed: int = 0) -> np.ndarray:
    try:
        from sklearn.cluster import KMeans
    except Exception:
        signed = 2 * labels - 1
        return _normalize(np.mean(x * signed[:, None], axis=0))
    model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    assignments = model.fit_predict(np.asarray(x, dtype=float))
    centers = model.cluster_centers_
    if n_clusters == 2:
        direction = centers[1] - centers[0]
    else:
        label_means = [float(np.mean(labels[assignments == i])) if np.any(assignments == i) else 0.0 for i in range(n_clusters)]
        high = int(np.argmax(label_means))
        low = int(np.argmin(label_means))
        direction = centers[high] - centers[low]
    if _label_accuracy(x, labels, direction) < _label_accuracy(x, labels, -direction):
        direction = -direction
    return _normalize(direction)


def acdc_single_ablation_score(y_full: float, y_without_component: float) -> float:
    return float(max(0.0, y_full - y_without_component))


def acdc_redundancy_failure_demo() -> dict[str, float | bool]:
    y00, y10, y01, y11 = 0.0, 1.0, 1.0, 1.0
    drop_a = acdc_single_ablation_score(y11, y01)
    drop_b = acdc_single_ablation_score(y11, y10)
    dual_drop = acdc_single_ablation_score(y11, y00)
    return {
        "single_drop_a": drop_a,
        "single_drop_b": drop_b,
        "dual_drop": dual_drop,
        "misses_redundancy": bool(max(drop_a, drop_b) == 0.0 and dual_drop > 0.0),
    }


def _reconstruction_error_rank1(x: np.ndarray, direction: np.ndarray) -> float:
    direction = _normalize(direction)
    centered = x - np.mean(x, axis=0, keepdims=True)
    recon = np.outer(centered @ direction, direction) + np.mean(x, axis=0, keepdims=True)
    return float(np.mean((x - recon) ** 2))


def score_dictionary_direction(name: str, x: np.ndarray, labels: np.ndarray, output: np.ndarray, direction: np.ndarray) -> DictionaryBaselineRow:
    label_accuracy = _label_accuracy(x, labels, direction)
    causal_use = _causal_use_score(x, output, direction)
    return DictionaryBaselineRow(
        name=name,
        label_accuracy=label_accuracy,
        causal_use_score=causal_use,
        reconstruction_error=_reconstruction_error_rank1(x, direction),
        false_mechanism=bool(label_accuracy >= 0.9 and causal_use < 0.2),
    )


def compare_dictionary_baselines(seed: int = 0) -> dict[str, object]:
    out: dict[str, object] = {"acdc_redundancy_failure": acdc_redundancy_failure_demo()}
    for setting, causal_output in (("random_labelable", False), ("trained_used", True)):
        x, labels, output = make_labelability_control(causal_output=causal_output, seed=seed)
        rows = [
            score_dictionary_direction("ica_first", x, labels, output, ica_first_direction(x, labels, seed=seed)).as_dict(),
            score_dictionary_direction(
                "kmeans_dictionary",
                x,
                labels,
                output,
                kmeans_dictionary_direction(x, labels, seed=seed),
            ).as_dict(),
        ]
        out[setting] = {
            "rows": rows,
            "mean_label_accuracy": float(np.mean([row["label_accuracy"] for row in rows])),
            "mean_causal_use_score": float(np.mean([row["causal_use_score"] for row in rows])),
            "false_mechanism_rate": float(np.mean([row["false_mechanism"] for row in rows])),
        }
    return out
