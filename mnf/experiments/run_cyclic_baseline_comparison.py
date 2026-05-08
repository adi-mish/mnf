from __future__ import annotations

import json
from collections.abc import Sequence

import numpy as np

from mnf.baselines.cyclic import compare_cyclic_baselines


def run(
    seed: int = 0,
    n: int = 3000,
    noise_levels: Sequence[float] = (0.0, 0.03, 0.08, 0.15, 0.3),
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
) -> dict[str, object]:
    rows = []
    for noise in noise_levels:
        metrics = [compare_cyclic_baselines(n=n, noise=noise, seed=seed + offset) for offset in seeds]
        row = {"noise": float(noise)}
        for key in metrics[0]:
            row[key] = float(np.mean([m[key] for m in metrics]))
        rows.append(row)
    return {"rows": rows}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
