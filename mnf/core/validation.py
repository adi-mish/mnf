from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import numpy as np

from mnf.core.causal_program import CausalProgram
from mnf.core.interventions import Intervention
from mnf.core.metrics import mse
from mnf.core.types import BinarySpace, CategoricalSpace, CyclicSpace, StateSpace


@dataclass(frozen=True)
class CircuitValidation:
    sufficiency_error: float
    necessity_effect: float
    kept_nodes: tuple[str, ...]
    ablated_nodes: tuple[str, ...]


@dataclass(frozen=True)
class EdgeInterventionValidation:
    edge: tuple[str, str]
    direct_effect: float
    passes: bool


def validate_node_set(
    program: CausalProgram,
    inputs: Sequence[Mapping[str, Any]],
    keep_nodes: Sequence[str],
    output: str | None = None,
) -> CircuitValidation:
    """Cheap necessity/sufficiency-style validation for causal programs.

    Sufficiency proxy: ablate everything outside keep_nodes and compare output.
    Necessity proxy: ablate keep_nodes and measure output change.

    This is not a substitute for model-level patching.  It is a ground-truth
    benchmark utility for executable causal programs.
    """
    output = output or program.outputs[0]
    keep = set(keep_nodes)
    all_nodes = set(program.node_names)
    ablate_outside = {n: Intervention(n, "ablate") for n in all_nodes - keep if n != output}
    ablate_inside = {n: Intervention(n, "ablate") for n in keep if n != output}
    base = np.array([program.run(x)[output] for x in inputs], dtype=float)
    suff = np.array([program.run(x, ablate_outside)[output] for x in inputs], dtype=float)
    nec = np.array([program.run(x, ablate_inside)[output] for x in inputs], dtype=float)
    return CircuitValidation(
        sufficiency_error=mse(base, suff),
        necessity_effect=mse(base, nec),
        kept_nodes=tuple(keep_nodes),
        ablated_nodes=tuple(sorted(all_nodes - keep)),
    )


def _counterfactual_value(space: StateSpace, current: Any, delta: float) -> Any:
    if isinstance(space, BinarySpace):
        return 0 if bool(current) else 1
    if isinstance(space, CyclicSpace):
        return space.project(space.index(current) + int(round(delta)))
    if isinstance(space, CategoricalSpace):
        idx = space.categories.index(current)
        return space.categories[(idx + 1) % len(space.categories)]
    try:
        return float(current) + delta
    except (TypeError, ValueError):
        return current


def validate_direct_edge_effects(
    program: CausalProgram,
    inputs: Sequence[Mapping[str, Any]],
    candidate_edges: Sequence[tuple[str, str]] | set[tuple[str, str]],
    effect_threshold: float = 1e-9,
    source_delta: float = 1.0,
) -> dict[tuple[str, str], EdgeInterventionValidation]:
    """Estimate direct edge effects by path-blocking interventions.

    For each candidate `src -> dst`, this clamps every earlier non-source node
    before `dst` to its baseline value, then counterfactually changes `src`.
    A transitive edge should disappear when its intermediate path is blocked,
    while a direct edge should still move `dst`.

    This is intended for executable ground-truth programs in MechanismLab.  It
    is the local analogue of activation/path patching for real models.
    """

    node_by_name = {node.name: node for node in program.nodes}
    order = {name: i for i, name in enumerate(program.node_names)}
    out: dict[tuple[str, str], EdgeInterventionValidation] = {}
    for edge in sorted(candidate_edges):
        src, dst = edge
        if src not in node_by_name or dst not in node_by_name or order[src] >= order[dst]:
            out[edge] = EdgeInterventionValidation(edge, 0.0, False)
            continue
        effects = []
        blockers = [name for name in program.node_names[: order[dst]] if name != src]
        for inp in inputs:
            base = program.run(inp)
            src_node = node_by_name[src]
            src_value = _counterfactual_value(src_node.state_space, base[src], source_delta)
            interventions = {name: Intervention(name, "clamp", base[name]) for name in blockers}
            interventions[src] = Intervention(src, "clamp", src_value)
            patched = program.run(inp, interventions)
            effects.append(node_by_name[dst].state_space.distance(base[dst], patched[dst]))
        direct_effect = float(np.mean(effects)) if effects else 0.0
        out[edge] = EdgeInterventionValidation(edge, direct_effect, direct_effect > effect_threshold)
    return out


def prune_edges_by_intervention(
    program: CausalProgram,
    inputs: Sequence[Mapping[str, Any]],
    candidate_edges: Sequence[tuple[str, str]] | set[tuple[str, str]],
    effect_threshold: float = 1e-9,
    source_delta: float = 1.0,
) -> set[tuple[str, str]]:
    validations = validate_direct_edge_effects(
        program,
        inputs,
        candidate_edges,
        effect_threshold=effect_threshold,
        source_delta=source_delta,
    )
    return {edge for edge, validation in validations.items() if validation.passes}
