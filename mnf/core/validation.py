from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import numpy as np

from mnf.core.causal_program import CausalProgram
from mnf.core.interventions import Intervention
from mnf.core.metrics import mse


@dataclass(frozen=True)
class CircuitValidation:
    sufficiency_error: float
    necessity_effect: float
    kept_nodes: tuple[str, ...]
    ablated_nodes: tuple[str, ...]


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
