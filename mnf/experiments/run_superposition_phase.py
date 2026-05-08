from __future__ import annotations

import json
import numpy as np

from mnf.benchmarks.superposition import SparseFeatureWorld, superposition_phase_statistic, predicted_superposition_favored


def run(seed: int = 0) -> dict[str, object]:
    rows = []
    for n_features in [8, 16, 32, 64]:
        for dim in [4, 8, 16]:
            for p in [0.02, 0.05, 0.15, 0.4]:
                world = SparseFeatureWorld.random(n_features, dim, p, seed=seed + n_features + dim)
                stat = superposition_phase_statistic(world)
                stat.update({
                    "n_features": n_features,
                    "activation_dim": dim,
                    "p": p,
                    "favored": predicted_superposition_favored(n_features, dim, p * p),
                })
                rows.append(stat)
    return {"rows": rows}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
