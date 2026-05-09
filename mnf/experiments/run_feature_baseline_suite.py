from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from mnf.baselines import compare_feature_baselines


def run(seeds: Sequence[int] = (0, 1, 2, 3, 4)) -> dict[str, object]:
    rows = [compare_feature_baselines(seed=seed) for seed in seeds]
    random_false = [row["random_labelable"]["false_mechanism_rate"] for row in rows]
    trained_false = [row["trained_used"]["false_mechanism_rate"] for row in rows]
    return {
        "rows": rows,
        "n_seeds": len(rows),
        "mean_random_false_mechanism_rate": float(np.mean(random_false)),
        "mean_trained_false_mechanism_rate": float(np.mean(trained_false)),
        "mean_random_causal_use_score": float(
            np.mean([row["random_labelable"]["mean_causal_use_score"] for row in rows])
        ),
        "mean_trained_causal_use_score": float(
            np.mean([row["trained_used"]["mean_causal_use_score"] for row in rows])
        ),
    }
