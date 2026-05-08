from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence
import numpy as np


@dataclass
class HierarchicalCode:
    """Structured parent-child feature code.

    active_children maps each parent to children that imply the parent.  Encoding
    can be faithful (parent fires whenever any child fires) or absorption-style
    (parent fires only when no child absorbs it).
    """

    parents: tuple[str, ...]
    children: Mapping[str, tuple[str, ...]]

    @property
    def features(self) -> tuple[str, ...]:
        out = list(self.parents)
        for p in self.parents:
            out.extend(self.children.get(p, ()))
        return tuple(out)

    def faithful_encode(self, active_leaf: str | None) -> dict[str, float]:
        z = {f: 0.0 for f in self.features}
        if active_leaf is None:
            return z
        for parent, children in self.children.items():
            if active_leaf in children:
                z[parent] = 1.0
                z[active_leaf] = 1.0
                return z
        if active_leaf in z:
            z[active_leaf] = 1.0
        return z

    def absorbed_encode(self, active_leaf: str | None) -> dict[str, float]:
        z = {f: 0.0 for f in self.features}
        if active_leaf is None:
            return z
        for _parent, children in self.children.items():
            if active_leaf in children:
                z[active_leaf] = 1.0
                return z
        if active_leaf in z:
            z[active_leaf] = 1.0
        return z


def absorption_score(parent_activation: np.ndarray, child_activation: np.ndarray, labels_parent: np.ndarray) -> float:
    """Measure a parent feature's failure to fire on true parent cases.

    Score 0 means parent activation is equally high on parent cases with and
    without child.  Score near 1 means child cases suppress/absorb the parent.
    """
    p = np.asarray(parent_activation, dtype=float)
    c = np.asarray(child_activation, dtype=float)
    y = np.asarray(labels_parent, dtype=bool)
    if p.shape != c.shape or p.shape != y.shape:
        raise ValueError("arrays must share shape")
    child_cases = y & (c > 0.5)
    parent_only = y & ~(c > 0.5)
    if not np.any(child_cases) or not np.any(parent_only):
        return 0.0
    mean_child = p[child_cases].mean()
    mean_parent = p[parent_only].mean()
    if mean_parent <= 1e-12:
        return 0.0
    return float(np.clip(1.0 - mean_child / mean_parent, 0.0, 1.0))


def simulate_absorbed_latents(
    n: int = 1000,
    child_rate: float = 0.2,
    parent_only_rate: float = 0.3,
    noise: float = 0.02,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Generate parent/child features with an absorption pathology."""
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    child = (u < child_rate).astype(float)
    parent_only = ((u >= child_rate) & (u < child_rate + parent_only_rate)).astype(float)
    parent_label = np.maximum(child, parent_only)
    faithful_parent = parent_label.copy()
    absorbed_parent = parent_only.copy()
    parent_latent = np.clip(absorbed_parent + noise * rng.normal(size=n), 0.0, None)
    child_latent = np.clip(child + noise * rng.normal(size=n), 0.0, None)
    return {
        "parent_label": parent_label,
        "child_label": child,
        "faithful_parent": faithful_parent,
        "absorbed_parent_latent": parent_latent,
        "child_latent": child_latent,
    }


def absorption_boundary(
    child_rate: float,
    sparsity_penalty: float,
    recon_cost_absorbed: float,
    recon_cost_faithful: float,
    interference_delta: float = 0.0,
) -> bool:
    """Return whether a simple MDL/rate-distortion model predicts absorption.

    Absorption is favored when the saved sparsity cost from not firing parent on
    child cases exceeds the added reconstruction/interference cost.
    """
    saved = sparsity_penalty * child_rate
    added = recon_cost_absorbed - recon_cost_faithful + interference_delta
    return bool(saved > added)
