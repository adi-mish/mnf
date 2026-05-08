from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from mnf.core.causal_program import CausalProgram, Node
from mnf.core.interventions import Intervention
from mnf.core.types import CategoricalSpace


SUBJECTS = ("alice", "bob", "carol", "dave")
RELATIONS = ("city", "instrument")
OBJECTS = ("paris", "rome", "london", "oslo", "violin", "piano", "flute", "drums")
DEFAULT_MEMORY = {
    ("alice", "city"): "paris",
    ("bob", "city"): "rome",
    ("carol", "city"): "london",
    ("dave", "city"): "oslo",
    ("alice", "instrument"): "violin",
    ("bob", "instrument"): "piano",
    ("carol", "instrument"): "flute",
    ("dave", "instrument"): "drums",
}


@dataclass(frozen=True)
class RelationalLookupSpec:
    subjects: tuple[str, ...] = SUBJECTS
    relations: tuple[str, ...] = RELATIONS
    objects: tuple[str, ...] = OBJECTS
    memory: dict[tuple[str, str], str] | None = None

    def resolved_memory(self) -> dict[tuple[str, str], str]:
        return dict(DEFAULT_MEMORY) if self.memory is None else dict(self.memory)


def make_relational_lookup_program(spec: RelationalLookupSpec | None = None) -> CausalProgram:
    """Subject-relation-object lookup with typed categorical variables."""

    spec = RelationalLookupSpec() if spec is None else spec
    memory = spec.resolved_memory()
    subject_space = CategoricalSpace(spec.subjects, name="subject")
    relation_space = CategoricalSpace(spec.relations, name="relation")
    object_space = CategoricalSpace(spec.objects, name="object")

    def subject_node(_state, inputs):
        return inputs["subject"]

    def relation_node(_state, inputs):
        return inputs["relation"]

    def object_node(state, _inputs):
        return memory[(state["subject"], state["relation"])]

    nodes = [
        Node("subject", (), subject_space, subject_node, "subject key"),
        Node("relation", (), relation_space, relation_node, "relation key"),
        Node("object", ("subject", "relation"), object_space, object_node, "lookup value"),
    ]
    return CausalProgram(nodes, outputs=("object",), name="relational_lookup")


def sample_relational_lookup_data(
    n: int = 256,
    seed: int = 0,
    spec: RelationalLookupSpec | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    spec = RelationalLookupSpec() if spec is None else spec
    rng = np.random.default_rng(seed)
    prog = make_relational_lookup_program(spec)
    inputs = [
        {
            "subject": str(rng.choice(spec.subjects)),
            "relation": str(rng.choice(spec.relations)),
        }
        for _ in range(n)
    ]
    states = [prog.run(inp) for inp in inputs]
    return inputs, states


def relation_swap_dataset(
    relation: str = "instrument",
    spec: RelationalLookupSpec | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, Intervention]], list[dict[str, str]]]:
    spec = RelationalLookupSpec() if spec is None else spec
    prog = make_relational_lookup_program(spec)
    inputs = [{"subject": subject, "relation": "city"} for subject in spec.subjects]
    interventions = [{"relation": Intervention("relation", "clamp", relation)} for _ in inputs]
    states = [prog.run(inp, intervention) for inp, intervention in zip(inputs, interventions)]
    return inputs, interventions, states
