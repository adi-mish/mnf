from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence
import math
import numpy as np


class StateSpace:
    """Abstract state space for a high-level causal variable.

    State spaces deliberately expose only two operations needed by MNF scoring:
    validation and a geometry-aware distance.  Gauge symmetries are handled by
    specific chart classes, not by the generic state space.
    """

    name: str = "state"

    def contains(self, value: Any) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def distance(self, a: Any, b: Any) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def project(self, value: Any) -> Any:
        """Return a valid nearby state when possible."""
        return value

    def dim(self) -> int | None:
        return None


@dataclass(frozen=True)
class ScalarSpace(StateSpace):
    low: float | None = None
    high: float | None = None
    name: str = "scalar"

    def contains(self, value: Any) -> bool:
        try:
            x = float(value)
        except (TypeError, ValueError):
            return False
        if self.low is not None and x < self.low:
            return False
        if self.high is not None and x > self.high:
            return False
        return True

    def project(self, value: Any) -> float:
        x = float(value)
        if self.low is not None:
            x = max(self.low, x)
        if self.high is not None:
            x = min(self.high, x)
        return x

    def distance(self, a: Any, b: Any) -> float:
        return abs(float(a) - float(b))

    def dim(self) -> int:
        return 1


@dataclass(frozen=True)
class BinarySpace(StateSpace):
    name: str = "binary"

    def contains(self, value: Any) -> bool:
        return value in (0, 1, False, True)

    def project(self, value: Any) -> int:
        return int(float(value) >= 0.5)

    def distance(self, a: Any, b: Any) -> float:
        return 0.0 if bool(a) == bool(b) else 1.0

    def dim(self) -> int:
        return 1


@dataclass(frozen=True)
class CategoricalSpace(StateSpace):
    categories: tuple[Any, ...]
    name: str = "categorical"

    def __init__(self, categories: Sequence[Any], name: str = "categorical"):
        if len(categories) == 0:
            raise ValueError("CategoricalSpace requires at least one category")
        object.__setattr__(self, "categories", tuple(categories))
        object.__setattr__(self, "name", name)

    def contains(self, value: Any) -> bool:
        return value in self.categories

    def project(self, value: Any) -> Any:
        if value in self.categories:
            return value
        if isinstance(value, (int, np.integer)):
            return self.categories[int(value) % len(self.categories)]
        raise ValueError(f"Cannot project {value!r} into categories {self.categories!r}")

    def distance(self, a: Any, b: Any) -> float:
        return 0.0 if a == b else 1.0

    def dim(self) -> int:
        return len(self.categories)


@dataclass(frozen=True)
class CyclicSpace(StateSpace):
    """Finite cyclic space C_n, represented internally by integers mod period.

    The circular distance is normalized to [0, 1].  For period=7, adjacent days
    have distance 1/3 because the maximum shortest-path distance is 3.
    """

    period: int
    labels: tuple[Any, ...] | None = None
    name: str = "cyclic"

    def __post_init__(self) -> None:
        if self.period < 2:
            raise ValueError("period must be >= 2")
        if self.labels is not None and len(self.labels) != self.period:
            raise ValueError("labels length must equal period")

    def index(self, value: Any) -> int:
        if self.labels is not None and value in self.labels:
            return self.labels.index(value)
        return int(value) % self.period

    def contains(self, value: Any) -> bool:
        try:
            self.index(value)
            return True
        except Exception:
            return False

    def project(self, value: Any) -> int | Any:
        idx = self.index(value)
        if self.labels is not None:
            return self.labels[idx]
        return idx

    def distance(self, a: Any, b: Any) -> float:
        ia, ib = self.index(a), self.index(b)
        raw = abs(ia - ib) % self.period
        circ = min(raw, self.period - raw)
        max_dist = self.period // 2
        return float(circ / max_dist) if max_dist > 0 else 0.0

    def angle(self, value: Any) -> float:
        return 2.0 * math.pi * self.index(value) / self.period

    def embed2(self, value: Any, radius: float = 1.0) -> np.ndarray:
        th = self.angle(value)
        return radius * np.array([math.cos(th), math.sin(th)], dtype=float)

    def nearest_from_angle(self, theta: float) -> int | Any:
        idx = int(round((theta % (2.0 * math.pi)) * self.period / (2.0 * math.pi))) % self.period
        return self.labels[idx] if self.labels is not None else idx

    def dim(self) -> int:
        return 2


@dataclass(frozen=True)
class VectorSpace(StateSpace):
    dimension: int
    norm: str = "l2"
    name: str = "vector"

    def __post_init__(self) -> None:
        if self.dimension < 1:
            raise ValueError("dimension must be positive")
        if self.norm not in {"l1", "l2", "linf"}:
            raise ValueError("norm must be one of 'l1', 'l2', 'linf'")

    def contains(self, value: Any) -> bool:
        arr = np.asarray(value, dtype=float)
        return arr.shape == (self.dimension,) and np.all(np.isfinite(arr))

    def project(self, value: Any) -> np.ndarray:
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size != self.dimension:
            raise ValueError(f"expected vector length {self.dimension}, got {arr.size}")
        return arr

    def distance(self, a: Any, b: Any) -> float:
        diff = self.project(a) - self.project(b)
        if self.norm == "l1":
            return float(np.abs(diff).sum())
        if self.norm == "linf":
            return float(np.abs(diff).max())
        return float(np.linalg.norm(diff))

    def dim(self) -> int:
        return self.dimension


@dataclass(frozen=True)
class ProductSpace(StateSpace):
    spaces: Mapping[str, StateSpace]
    name: str = "product"

    def contains(self, value: Any) -> bool:
        if not isinstance(value, Mapping):
            return False
        return all(k in value and s.contains(value[k]) for k, s in self.spaces.items())

    def project(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return {k: s.project(value[k]) for k, s in self.spaces.items()}

    def distance(self, a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
        terms = [s.distance(a[k], b[k]) ** 2 for k, s in self.spaces.items()]
        return float(math.sqrt(sum(terms)))

    def dim(self) -> int | None:
        dims = [s.dim() for s in self.spaces.values()]
        if any(d is None for d in dims):
            return None
        return int(sum(dims))


@dataclass(frozen=True)
class Hierarchy:
    """A tiny helper for parent-child semantic hierarchies."""

    parent_to_children: Mapping[str, tuple[str, ...]]

    def ancestors(self, node: str) -> set[str]:
        out: set[str] = set()
        changed = True
        while changed:
            changed = False
            for parent, children in self.parent_to_children.items():
                if node in children or any(c in out for c in children):
                    if parent not in out:
                        out.add(parent)
                        changed = True
        return out

    def descendants(self, node: str) -> set[str]:
        out: set[str] = set()
        stack = list(self.parent_to_children.get(node, ()))
        while stack:
            cur = stack.pop()
            if cur not in out:
                out.add(cur)
                stack.extend(self.parent_to_children.get(cur, ()))
        return out

    def implies(self, child: str, parent: str) -> bool:
        return parent in self.ancestors(child)
