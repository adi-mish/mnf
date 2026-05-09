from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np


ChartRead = Callable[[Mapping[str, object]], Mapping[str, float]]


@dataclass(frozen=True)
class MechanismChart:
    name: str
    atom_names: tuple[str, ...]
    read: ChartRead
    domain_label: str = "default"

    def __init__(
        self,
        name: str,
        atom_names: Sequence[str],
        read: ChartRead,
        domain_label: str = "default",
    ):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "atom_names", tuple(atom_names))
        object.__setattr__(self, "read", read)
        object.__setattr__(self, "domain_label", domain_label)

    def encode(self, sample: Mapping[str, object]) -> dict[str, float]:
        values = {key: float(value) for key, value in self.read(sample).items()}
        missing = [name for name in self.atom_names if name not in values]
        if missing:
            raise ValueError(f"chart {self.name!r} missing atoms {missing!r}")
        return {name: values[name] for name in self.atom_names}


def state_distance(a: Mapping[str, float], b: Mapping[str, float], keys: Sequence[str] | None = None) -> float:
    keys = tuple(keys or sorted(set(a) & set(b)))
    if not keys:
        return 0.0
    va = np.asarray([float(a[key]) for key in keys], dtype=float)
    vb = np.asarray([float(b[key]) for key in keys], dtype=float)
    return float(np.mean((va - vb) ** 2))
