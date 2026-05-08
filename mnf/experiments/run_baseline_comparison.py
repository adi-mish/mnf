from __future__ import annotations

import json

from mnf.baselines.selection import compare_shortcut_selectors


def run(seed: int = 0) -> dict[str, object]:
    return {
        "shortcut_selector_comparison": compare_shortcut_selectors(seed=seed),
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
