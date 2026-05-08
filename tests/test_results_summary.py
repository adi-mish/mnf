import json
from pathlib import Path

from mnf.experiments.results_summary import summarize_research_sweeps


def test_results_summary_mentions_core_experiments():
    data = json.loads(Path("docs/research_sweeps.json").read_text())
    text = summarize_research_sweeps(data)
    assert "Ground Truth Recovery" in text
    assert "Shortcut Baseline" in text
    assert "Induction Match-Copy" in text
    if "interaction_suite" in data:
        assert "Mechanism Interactions" in text
        assert "Interaction recovery F1" in text
    assert "Transition Atoms" in text
