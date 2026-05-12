from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mnf.models.tiny_transformer import modular_addition_dataset


@dataclass(frozen=True)
class TinyTask:
    name: str
    modulus: int
    inputs: Any
    targets: Any
    state_space: str

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "modulus": self.modulus, "state_space": self.state_space, "n_examples": len(self.targets)}


def modular_addition_task(modulus: int = 7, device: str = "cpu") -> TinyTask:
    x, y = modular_addition_dataset(modulus=modulus, device=device)
    return TinyTask(
        name=f"modular_addition_C{modulus}",
        modulus=modulus,
        inputs=x,
        targets=y,
        state_space=f"C_{modulus}",
    )
