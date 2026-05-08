from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from mnf.core.causal_program import CausalProgram


@dataclass(frozen=True)
class MechanismStep:
    op: str
    target: str
    sources: tuple[str, ...] = ()
    description: str = ""
    tags: tuple[str, ...] = ()

    def to_line(self) -> str:
        src = ",".join(self.sources) if self.sources else "input"
        tags = f" [{','.join(self.tags)}]" if self.tags else ""
        desc = f" -- {self.description}" if self.description else ""
        return f"{self.op} {src} -> {self.target}{tags}{desc}"


@dataclass
class MechanismDSL:
    """Small text representation of an executable/intervention-ready mechanism."""

    name: str
    steps: list[MechanismStep] = field(default_factory=list)
    outputs: tuple[str, ...] = ()

    @classmethod
    def from_program(cls, program: CausalProgram) -> "MechanismDSL":
        steps = []
        for node in program.nodes:
            op = "source" if not node.parents else "compute"
            steps.append(
                MechanismStep(
                    op=op,
                    target=node.name,
                    sources=node.parents,
                    description=node.description,
                    tags=(node.state_space.name,),
                )
            )
        return cls(name=program.name, steps=steps, outputs=program.outputs)

    def to_text(self) -> str:
        lines = [f"mechanism {self.name}"]
        for step in self.steps:
            lines.append(f"  {step.to_line()}")
        if self.outputs:
            lines.append(f"outputs: {', '.join(self.outputs)}")
        return "\n".join(lines)

    def variables(self) -> set[str]:
        out = set()
        for step in self.steps:
            out.add(step.target)
            out.update(step.sources)
        out.discard("input")
        return out
