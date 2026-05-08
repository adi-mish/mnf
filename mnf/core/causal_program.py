from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence
import hashlib
import inspect

from mnf.core.types import StateSpace
from mnf.core.interventions import Intervention


NodeFunction = Callable[[Mapping[str, Any], Mapping[str, Any]], Any]


@dataclass
class Node:
    """A typed structural equation in a high-level causal program.

    The function receives the partial state computed so far and the external
    inputs.  Parents are used for validation, graph recovery metrics, and MDL.
    """

    name: str
    parents: tuple[str, ...]
    state_space: StateSpace
    func: NodeFunction
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def compute(self, state: Mapping[str, Any], inputs: Mapping[str, Any]) -> Any:
        value = self.func(state, inputs)
        if not self.state_space.contains(value):
            value = self.state_space.project(value)
        return value


@dataclass
class CausalProgram:
    """Executable acyclic high-level causal program."""

    nodes: list[Node]
    outputs: tuple[str, ...]
    name: str = "causal_program"

    def __post_init__(self) -> None:
        names = [n.name for n in self.nodes]
        if len(names) != len(set(names)):
            raise ValueError("Node names must be unique")
        self._node_by_name = {n.name: n for n in self.nodes}
        self._validate_order()
        for out in self.outputs:
            if out not in self._node_by_name:
                raise ValueError(f"Unknown output node {out!r}")

    @property
    def node_names(self) -> tuple[str, ...]:
        return tuple(n.name for n in self.nodes)

    @property
    def edges(self) -> set[tuple[str, str]]:
        return {(p, n.name) for n in self.nodes for p in n.parents}

    def _validate_order(self) -> None:
        seen: set[str] = set()
        for node in self.nodes:
            missing = [p for p in node.parents if p not in seen]
            if missing:
                raise ValueError(
                    f"Node {node.name!r} has parents {missing!r} that are not earlier nodes"
                )
            seen.add(node.name)

    def run(
        self,
        inputs: Mapping[str, Any] | None = None,
        interventions: Mapping[str, Intervention] | None = None,
        return_all: bool = True,
    ) -> dict[str, Any]:
        inputs = {} if inputs is None else dict(inputs)
        interventions = {} if interventions is None else dict(interventions)
        state: dict[str, Any] = {}
        for node in self.nodes:
            if node.name in interventions and interventions[node.name].kind == "clamp":
                value = interventions[node.name].value
                if not node.state_space.contains(value):
                    value = node.state_space.project(value)
                state[node.name] = value
                continue
            value = node.compute(state, inputs)
            if node.name in interventions:
                value = interventions[node.name].apply(value)
                if not node.state_space.contains(value):
                    value = node.state_space.project(value)
            state[node.name] = value
        if return_all:
            return state
        return {k: state[k] for k in self.outputs}

    def intervene(self, variable: str, kind: str, value: Any = None) -> dict[str, Intervention]:
        if variable not in self._node_by_name:
            raise ValueError(f"Unknown variable {variable!r}")
        return {variable: Intervention(variable, kind, value)}

    def subprogram(self, keep: Sequence[str], name: str | None = None) -> "CausalProgram":
        keep_set = set(keep)
        nodes = [n for n in self.nodes if n.name in keep_set and all(p in keep_set for p in n.parents)]
        outputs = tuple(o for o in self.outputs if o in keep_set)
        return CausalProgram(nodes, outputs, name=name or f"{self.name}_sub")

    def signature(self) -> str:
        """Stable-ish hash of the graph and function sources for provenance."""
        h = hashlib.sha256()
        h.update(self.name.encode())
        for node in self.nodes:
            h.update(node.name.encode())
            h.update(repr(node.parents).encode())
            h.update(node.state_space.name.encode())
            try:
                h.update(inspect.getsource(node.func).encode())
            except (OSError, TypeError):
                h.update(repr(node.func).encode())
        return h.hexdigest()[:16]

    def describe(self) -> str:
        lines = [f"CausalProgram({self.name}, outputs={self.outputs}, sig={self.signature()})"]
        for node in self.nodes:
            parents = ", ".join(node.parents) or "<input>"
            desc = f" -- {node.description}" if node.description else ""
            lines.append(f"  {parents} -> {node.name} : {node.state_space.name}{desc}")
        return "\n".join(lines)
