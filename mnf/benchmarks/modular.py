from __future__ import annotations

import numpy as np

from mnf.core.causal_program import CausalProgram, Node
from mnf.core.types import CyclicSpace


def make_modular_addition_program(period: int = 7) -> CausalProgram:
    """Typed cyclic program for modular addition in C_period."""

    space = CyclicSpace(period)

    def a_node(_state, inputs):
        return space.index(inputs["a"])

    def b_node(_state, inputs):
        return space.index(inputs["b"])

    def sum_node(state, _inputs):
        return (space.index(state["a"]) + space.index(state["b"])) % period

    nodes = [
        Node("a", (), space, a_node, "left cyclic addend"),
        Node("b", (), space, b_node, "right cyclic addend"),
        Node("sum", ("a", "b"), space, sum_node, "modular sum"),
    ]
    return CausalProgram(nodes, outputs=("sum",), name=f"modular_addition_C_{period}")


def sample_modular_addition_data(
    n: int = 256,
    period: int = 7,
    seed: int = 0,
) -> tuple[list[dict[str, int]], list[dict[str, int]]]:
    rng = np.random.default_rng(seed)
    prog = make_modular_addition_program(period)
    inputs = [
        {"a": int(a), "b": int(b)}
        for a, b in rng.integers(0, period, size=(n, 2))
    ]
    states = [prog.run(inp) for inp in inputs]
    return inputs, states


def modular_shift_error(period: int = 7, shift: int = 2) -> float:
    """Return cyclic prediction error after shifting one addend."""

    prog = make_modular_addition_program(period)
    space = CyclicSpace(period)
    errs = []
    for a in range(period):
        for b in range(period):
            base = prog.run({"a": a, "b": b})
            shifted = prog.run({"a": (a + shift) % period, "b": b})
            expected = (base["sum"] + shift) % period
            errs.append(space.distance(shifted["sum"], expected))
    return float(np.mean(errs))
