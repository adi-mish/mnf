from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mnf.experiments.results_summary import summarize_research_sweeps


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / "docs" / "research_sweeps.json").read_text())
    summary = summarize_research_sweeps(data)
    (root / "docs" / "RESULTS_SUMMARY.md").write_text(summary + "\n")


if __name__ == "__main__":
    main()
