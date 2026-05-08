from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from mnf.benchmarks.shortcut import CandidateInvarianceScore, make_shortcut_dataset, score_shortcut_candidates


@dataclass(frozen=True)
class SelectorResult:
    selector: str
    selected: str
    train_accuracy: float
    shifted_accuracy: float
    invariance_gap: float


def _as_result(selector: str, score: CandidateInvarianceScore) -> SelectorResult:
    return SelectorResult(
        selector=selector,
        selected=score.name,
        train_accuracy=score.train_accuracy,
        shifted_accuracy=score.shifted_accuracy,
        invariance_gap=score.invariance_gap,
    )


def select_by_train_accuracy(scores: Mapping[str, CandidateInvarianceScore]) -> SelectorResult:
    """Labelability-style selector that only sees the training environment."""

    score = max(scores.values(), key=lambda s: s.train_accuracy)
    return _as_result("train_accuracy", score)


def select_by_invariance(scores: Mapping[str, CandidateInvarianceScore]) -> SelectorResult:
    """MNF-style selector using environment stability before train fit."""

    accepted = [score for score in scores.values() if score.accepted]
    if accepted:
        score = max(accepted, key=lambda s: (s.min_environment_accuracy, s.train_accuracy))
    else:
        score = min(scores.values(), key=lambda s: (s.invariance_gap, -s.min_environment_accuracy))
    return _as_result("invariance", score)


def compare_shortcut_selectors(seed: int = 0) -> dict[str, object]:
    """Compare train-only and invariance-aware selection on a hard shortcut case."""

    dataset = make_shortcut_dataset(
        n_per_environment=2000,
        causal_strength=0.35,
        shortcut_strength=1.0,
        noise=0.2,
        seed=seed,
    )
    scores = score_shortcut_candidates(dataset)
    train_only = select_by_train_accuracy(scores)
    invariant = select_by_invariance(scores)
    return {
        "candidate_scores": {
            name: {
                "train_accuracy": score.train_accuracy,
                "shifted_accuracy": score.shifted_accuracy,
                "invariance_gap": score.invariance_gap,
                "accepted": score.accepted,
            }
            for name, score in scores.items()
        },
        "train_only_selector": train_only.__dict__,
        "invariance_selector": invariant.__dict__,
    }
