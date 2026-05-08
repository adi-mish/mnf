from __future__ import annotations

from typing import Iterable, Mapping, Sequence
import itertools
import numpy as np

from mnf.core.metrics import correlation_matrix


def discover_linear_effect_graph(
    states: Mapping[str, np.ndarray],
    threshold: float = 0.2,
    forbid_backward: bool = True,
) -> set[tuple[str, str]]:
    """Infer candidate edges using lag-free linear effects/correlation.

    This is a cheap proposal mechanism, not proof of causality.  The MNF pipeline
    is supposed to validate proposed edges by interventions afterward.
    """
    names = list(states.keys())
    edges: set[tuple[str, str]] = set()
    for i, src in enumerate(names):
        for j, dst in enumerate(names):
            if src == dst:
                continue
            if forbid_backward and j <= i:
                continue
            x = np.asarray(states[src], dtype=float).reshape(len(states[src]), -1)
            y = np.asarray(states[dst], dtype=float).reshape(len(states[dst]), -1)
            corr = correlation_matrix(x, y)
            effect = float(np.max(np.abs(corr)))
            if effect >= threshold:
                edges.add((src, dst))
    return edges


def prune_edges_by_mdl(
    edges: Iterable[tuple[str, str]],
    edge_scores: Mapping[tuple[str, str], float],
    penalty: float = 0.05,
) -> set[tuple[str, str]]:
    """Keep edges whose score exceeds a description-length penalty."""
    return {e for e in edges if edge_scores.get(e, 0.0) > penalty}


def edge_effect_scores(states: Mapping[str, np.ndarray], edges: Iterable[tuple[str, str]]) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    for src, dst in edges:
        x = np.asarray(states[src], dtype=float).reshape(len(states[src]), -1)
        y = np.asarray(states[dst], dtype=float).reshape(len(states[dst]), -1)
        out[(src, dst)] = float(np.max(np.abs(correlation_matrix(x, y))))
    return out
