from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import numpy as np


class AtomKind(str, Enum):
    STATE = "state"
    TRANSITION = "transition"
    ROUTE = "route"
    GATE = "gate"
    MEMORY = "memory"


@dataclass
class CandidateAtom:
    name: str
    kind: AtomKind
    activations: np.ndarray | None = None
    score: float = 0.0
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def description_length(self) -> float:
        base = len(self.name) / 16.0 + len(self.description) / 64.0 + 1.0
        if self.activations is not None:
            base += self.activations.shape[-1] / 128.0 if self.activations.ndim > 1 else 1.0 / 128.0
        return float(base)
