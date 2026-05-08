from __future__ import annotations

from itertools import product
from typing import Sequence
import numpy as np

from mnf.core.causal_program import CausalProgram, Node
from mnf.core.interventions import Intervention
from mnf.core.types import BinarySpace


def make_gated_xor_program() -> CausalProgram:
    """Ground-truth nonlinear gated mechanism.

    The program separates a represented intermediate state, `xor`, from a
    policy/control gate.  It is a small MechanismLab target for methods that
    must recover nonlinear state atoms and a gate-to-output causal role.
    """

    def x1_node(_state, inputs):
        return int(inputs["x1"])

    def x2_node(_state, inputs):
        return int(inputs["x2"])

    def gate_node(_state, inputs):
        return int(inputs["gate"])

    def xor_node(state, _inputs):
        return int(bool(state["x1"]) ^ bool(state["x2"]))

    def y_node(state, _inputs):
        return int(bool(state["gate"]) and bool(state["xor"]))

    nodes = [
        Node("x1", (), BinarySpace(), x1_node, "input bit"),
        Node("x2", (), BinarySpace(), x2_node, "input bit"),
        Node("gate", (), BinarySpace(), gate_node, "control gate"),
        Node("xor", ("x1", "x2"), BinarySpace(), xor_node, "nonlinear parity state"),
        Node("y", ("gate", "xor"), BinarySpace(), y_node, "gated output"),
    ]
    return CausalProgram(nodes, outputs=("y",), name="gated_xor")


def exhaustive_gated_xor_inputs() -> list[dict[str, int]]:
    return [
        {"x1": int(x1), "x2": int(x2), "gate": int(gate)}
        for x1, x2, gate in product((0, 1), repeat=3)
    ]


def sample_gated_xor_data(n: int = 256, seed: int = 0) -> tuple[list[dict[str, int]], list[dict[str, int]]]:
    rng = np.random.default_rng(seed)
    prog = make_gated_xor_program()
    inputs = [
        {"x1": int(x1), "x2": int(x2), "gate": int(gate)}
        for x1, x2, gate in rng.integers(0, 2, size=(n, 3))
    ]
    states = [prog.run(inp) for inp in inputs]
    return inputs, states


def gated_xor_intervention_dataset(
    inputs: Sequence[dict[str, int]] | None = None,
) -> tuple[list[dict[str, int]], list[dict[str, Intervention]], list[dict[str, int]]]:
    prog = make_gated_xor_program()
    base_inputs = exhaustive_gated_xor_inputs() if inputs is None else list(inputs)
    interventions = []
    states = []
    for inp in base_inputs:
        target_gate = 1 - int(inp["gate"])
        intervention = {"gate": Intervention("gate", "clamp", target_gate)}
        interventions.append(intervention)
        states.append(prog.run(inp, intervention))
    return base_inputs, interventions, states
