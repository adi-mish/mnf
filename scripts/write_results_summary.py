from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mnf.experiments.results_summary import summarize_research_sweeps
from mnf.certificates.pareto_plot import certificate_pareto_svg


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / "docs" / "research_sweeps.json").read_text())
    summary = summarize_research_sweeps(data)
    (root / "docs" / "RESULTS_SUMMARY.md").write_text(summary + "\n")
    if "certificate_demo" in data:
        certificate_pareto_svg(data["certificate_demo"]["report"], root / "docs" / "certificate_pareto.svg")


if __name__ == "__main__":
    main()
