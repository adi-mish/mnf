from __future__ import annotations

from typing import Any
from mnf.core.causal_program import CausalProgram
from mnf.core.metrics import description_length_from_counts


def state_dim_total(program: CausalProgram) -> int:
    total = 0
    for node in program.nodes:
        dim = node.state_space.dim()
        total += 1 if dim is None else dim
    return total


def program_description_length(program: CausalProgram, n_params: int = 0) -> float:
    return description_length_from_counts(
        n_nodes=len(program.nodes),
        n_edges=len(program.edges),
        n_state_dims=state_dim_total(program),
        n_params=n_params,
    )


def object_description_length(obj: Any) -> float:
    """Crude fallback MDL proxy based on repr length.

    This is not intended as a final MDL theory.  It makes the scaffold executable
    and gives a consistent penalty for high-complexity mappings.
    """
    return len(repr(obj).encode("utf-8")) / 64.0
