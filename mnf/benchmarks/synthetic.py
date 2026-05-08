from __future__ import annotations

from typing import Any
import numpy as np

from mnf.core.causal_program import CausalProgram, Node
from mnf.core.types import ScalarSpace
from mnf.core.interventions import Intervention


def make_chain_program(noise_free: bool = True) -> CausalProgram:
    """Ground-truth scalar chain X -> A -> B -> Y.

    Inputs provide x.  This is a simple testbed for commutation and edge
    recovery: interventions on A should change B and Y but not X.
    """

    def x_node(_state, inputs):
        return float(inputs["x"])

    def a_node(state, _inputs):
        return 2.0 * state["x"] + 1.0

    def b_node(state, _inputs):
        return -0.5 * state["a"] + 0.25

    def y_node(state, _inputs):
        return state["b"] ** 2

    nodes = [
        Node("x", (), ScalarSpace(), x_node, "input scalar"),
        Node("a", ("x",), ScalarSpace(), a_node, "linear hidden state"),
        Node("b", ("a",), ScalarSpace(), b_node, "linear hidden state"),
        Node("y", ("b",), ScalarSpace(low=0.0), y_node, "observable output"),
    ]
    return CausalProgram(nodes, outputs=("y",), name="chain_x_a_b_y")


def sample_chain_data(n: int = 256, seed: int = 0) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    rng = np.random.default_rng(seed)
    prog = make_chain_program()
    inputs = [{"x": float(x)} for x in rng.normal(size=n)]
    states = [prog.run(inp) for inp in inputs]
    return inputs, states


def chain_intervention_dataset(n: int = 64, seed: int = 1) -> tuple[list[dict[str, float]], list[dict[str, Intervention]], list[dict[str, float]]]:
    rng = np.random.default_rng(seed)
    prog = make_chain_program()
    inputs = [{"x": float(x)} for x in rng.normal(size=n)]
    interventions = []
    states = []
    for inp in inputs:
        val = float(rng.normal())
        intervention = {"a": Intervention("a", "clamp", val)}
        interventions.append(intervention)
        states.append(prog.run(inp, intervention))
    return inputs, interventions, states
