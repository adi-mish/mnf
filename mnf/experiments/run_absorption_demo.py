from __future__ import annotations

import json
from mnf.benchmarks.hierarchy import make_hierarchy_dataset, compare_flat_vs_hierarchical_codes


def run(seed: int = 0) -> dict[str, float]:
    ds = make_hierarchy_dataset(seed=seed)
    return compare_flat_vs_hierarchical_codes(ds)


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
