from __future__ import annotations

import json
from collections.abc import Sequence

import numpy as np

from mnf.benchmarks.hierarchy import make_hierarchy_dataset, compare_flat_vs_hierarchical_codes


def _mean(metrics: list[dict[str, float]], key: str) -> float:
    return float(np.mean([m[key] for m in metrics]))


def run(
    seed: int = 0,
    n: int = 2000,
    child_rates: Sequence[float] = (0.05, 0.1, 0.2, 0.35, 0.5),
    parent_only_rates: Sequence[float] = (0.1, 0.3, 0.5),
    noise_levels: Sequence[float] = (0.01, 0.05, 0.1),
    seeds: Sequence[int] = (0, 1, 2),
) -> dict[str, object]:
    """CPU-only phase sweep for hierarchy absorption.

    The resulting rows quantify where flat parent features fail and where the
    explicit child-implies-parent correction restores parent recovery.
    """

    rows = []
    for child_rate in child_rates:
        for parent_only_rate in parent_only_rates:
            if child_rate + parent_only_rate >= 0.95:
                continue
            for noise in noise_levels:
                metrics = []
                for offset in seeds:
                    ds = make_hierarchy_dataset(
                        n=n,
                        child_rate=child_rate,
                        parent_only_rate=parent_only_rate,
                        noise=noise,
                        seed=seed + offset,
                    )
                    metrics.append(compare_flat_vs_hierarchical_codes(ds))
                flat_error = _mean(metrics, "flat_parent_error")
                hierarchical_error = _mean(metrics, "hierarchical_parent_error")
                rows.append(
                    {
                        "child_rate": float(child_rate),
                        "parent_only_rate": float(parent_only_rate),
                        "noise": float(noise),
                        "absorption_score": _mean(metrics, "absorption_score"),
                        "flat_parent_error": flat_error,
                        "hierarchical_parent_error": hierarchical_error,
                        "hierarchical_error_reduction": flat_error - hierarchical_error,
                    }
                )
    return {"rows": rows}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
