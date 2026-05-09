from __future__ import annotations

import json

from mnf.benchmarks.atlas import no_global_chart_metrics


def run() -> dict[str, object]:
    return {
        "no_global_chart": no_global_chart_metrics(),
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
