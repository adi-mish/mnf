from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence
import numpy as np


@dataclass(frozen=True)
class MemorizingAlignment:
    """High-capacity lookup map used as a vacuity/naturalness control."""

    table: Mapping[int, int]
    default: int = 0

    @classmethod
    def fit(cls, ids: Sequence[int], labels: Sequence[int]) -> "MemorizingAlignment":
        return cls(table={int(i): int(y) for i, y in zip(ids, labels)})

    def predict(self, ids: Sequence[int]) -> np.ndarray:
        return np.array([self.table.get(int(i), self.default) for i in ids], dtype=int)

    def description_length(self) -> float:
        return float(len(self.table))


def memorization_control(
    n_train: int = 512,
    n_test: int = 512,
    seed: int = 0,
) -> dict[str, float]:
    """Show how arbitrary alignments can fit observations without mechanism."""

    rng = np.random.default_rng(seed)
    train_ids = np.arange(n_train)
    test_ids = np.arange(n_train, n_train + n_test)
    train_labels = rng.integers(0, 2, size=n_train)
    test_labels = rng.integers(0, 2, size=n_test)
    align = MemorizingAlignment.fit(train_ids, train_labels)
    train_acc = float(np.mean(align.predict(train_ids) == train_labels))
    test_acc = float(np.mean(align.predict(test_ids) == test_labels))
    return {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "description_length": align.description_length(),
        "test_generalization_gap": train_acc - test_acc,
    }
