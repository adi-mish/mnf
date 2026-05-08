from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence
import numpy as np

from mnf.core.causal_program import CausalProgram, Node
from mnf.core.types import CategoricalSpace


@dataclass
class InductionDataset:
    sequences: np.ndarray
    labels: np.ndarray
    train_pairs: set[tuple[int, int]]
    test_pairs: set[tuple[int, int]]


def _all_ordered_pairs(vocab_size: int) -> list[tuple[int, int]]:
    return [(a, b) for a in range(vocab_size) for b in range(vocab_size) if a != b]


def _make_sequences(
    pairs: Sequence[tuple[int, int]],
    continuations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    labels = []
    for (a, b), c in zip(pairs, continuations):
        rows.append([a, b, int(c), a, b])
        labels.append(int(c))
    return np.asarray(rows, dtype=int), np.asarray(labels, dtype=int)


def make_induction_generalization_dataset(
    vocab_size: int = 16,
    n_train_pairs: int = 120,
    n_test_pairs: int = 80,
    seed: int = 0,
) -> InductionDataset:
    """Generate train/test sequences for an induction-style match-copy task."""

    all_pairs = _all_ordered_pairs(vocab_size)
    if n_train_pairs + n_test_pairs > len(all_pairs):
        raise ValueError("requested more pairs than the vocabulary supports")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(all_pairs))
    train_pairs = [all_pairs[i] for i in perm[:n_train_pairs]]
    test_pairs = [all_pairs[i] for i in perm[n_train_pairs:n_train_pairs + n_test_pairs]]
    train_c = rng.integers(0, vocab_size, size=len(train_pairs))
    test_c = rng.integers(0, vocab_size, size=len(test_pairs))
    train_x, train_y = _make_sequences(train_pairs, train_c)
    test_x, test_y = _make_sequences(test_pairs, test_c)
    return InductionDataset(
        sequences=np.concatenate([train_x, test_x], axis=0),
        labels=np.concatenate([train_y, test_y], axis=0),
        train_pairs=set(train_pairs),
        test_pairs=set(test_pairs),
    )


def induction_copy_predict(sequences: np.ndarray, no_match_value: int = 0) -> np.ndarray:
    """Return the token following the previous occurrence of the final bigram."""

    seqs = np.asarray(sequences, dtype=int)
    out = []
    for seq in seqs:
        query = tuple(seq[-2:])
        pred = no_match_value
        for pos in range(0, len(seq) - 2):
            if tuple(seq[pos:pos + 2]) == query:
                pred = int(seq[pos + 2])
                break
        out.append(pred)
    return np.asarray(out, dtype=int)


@dataclass
class PairMemorizer:
    table: Mapping[tuple[int, int], int]
    default: int

    @classmethod
    def fit(cls, sequences: np.ndarray, labels: np.ndarray) -> "PairMemorizer":
        buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
        for seq, label in zip(np.asarray(sequences, dtype=int), np.asarray(labels, dtype=int)):
            buckets[tuple(seq[-2:])].append(int(label))
        table = {
            pair: Counter(values).most_common(1)[0][0]
            for pair, values in buckets.items()
        }
        default = Counter(int(v) for v in labels).most_common(1)[0][0]
        return cls(table=table, default=default)

    def predict(self, sequences: np.ndarray) -> np.ndarray:
        return np.asarray(
            [self.table.get(tuple(seq[-2:]), self.default) for seq in np.asarray(sequences, dtype=int)],
            dtype=int,
        )

    def description_length(self) -> float:
        return float(len(self.table))


def make_induction_program(vocab_size: int = 16) -> CausalProgram:
    token_space = CategoricalSpace(tuple(range(vocab_size)), name="token")
    pos_space = CategoricalSpace((-1, 0, 1, 2), name="match_position")

    def make_token_node(i: int):
        return lambda _state, inputs, i=i: int(inputs["sequence"][i])

    def match_position(state, _inputs):
        query = (state["t3"], state["t4"])
        for pos in (0, 1, 2):
            if (state[f"t{pos}"], state[f"t{pos + 1}"]) == query:
                return pos
        return -1

    def copied_token(state, _inputs):
        pos = state["match_position"]
        if pos < 0:
            return 0
        return state[f"t{pos + 2}"]

    nodes = [
        Node(f"t{i}", (), token_space, make_token_node(i), f"sequence token {i}")
        for i in range(5)
    ]
    nodes.extend([
        Node("match_position", ("t0", "t1", "t2", "t3", "t4"), pos_space, match_position, "previous final-bigram match"),
        Node("copied_token", ("match_position", "t2", "t3", "t4"), token_space, copied_token, "token after matched bigram"),
        Node("y", ("copied_token",), token_space, lambda state, _inputs: state["copied_token"], "output token"),
    ])
    return CausalProgram(nodes, outputs=("y",), name="induction_match_copy")


def compare_induction_vs_memorization(seed: int = 0) -> dict[str, float]:
    ds = make_induction_generalization_dataset(seed=seed)
    n_train = len(ds.train_pairs)
    train_x, test_x = ds.sequences[:n_train], ds.sequences[n_train:]
    train_y, test_y = ds.labels[:n_train], ds.labels[n_train:]
    memorizer = PairMemorizer.fit(train_x, train_y)
    return {
        "memorizer_train_accuracy": float(np.mean(memorizer.predict(train_x) == train_y)),
        "memorizer_test_accuracy": float(np.mean(memorizer.predict(test_x) == test_y)),
        "copy_train_accuracy": float(np.mean(induction_copy_predict(train_x) == train_y)),
        "copy_test_accuracy": float(np.mean(induction_copy_predict(test_x) == test_y)),
        "memorizer_description_length": memorizer.description_length(),
        "copy_description_length": 4.0,
    }
