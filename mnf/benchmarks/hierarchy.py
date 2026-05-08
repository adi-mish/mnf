from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from mnf.charts.hierarchical import absorption_score, simulate_absorbed_latents


@dataclass
class HierarchyDataset:
    activations: np.ndarray
    parent_label: np.ndarray
    child_label: np.ndarray
    parent_direction: np.ndarray
    child_direction: np.ndarray
    absorbed_parent_direction: np.ndarray


def make_hierarchy_dataset(
    n: int = 1000,
    activation_dim: int = 16,
    child_rate: float = 0.2,
    parent_only_rate: float = 0.3,
    noise: float = 0.05,
    seed: int = 0,
) -> HierarchyDataset:
    """Generate activations where child implies parent.

    Faithful representation writes parent + child.  Absorbed representation only
    writes child on child cases and parent on parent-only cases.  The activation
    combines directions so downstream tests can measure the failure mode.
    """
    rng = np.random.default_rng(seed)
    latent = simulate_absorbed_latents(n, child_rate, parent_only_rate, noise=0.0, seed=seed)
    parent_dir = rng.normal(size=activation_dim)
    parent_dir /= np.linalg.norm(parent_dir)
    child_dir = rng.normal(size=activation_dim)
    child_dir -= parent_dir * float(child_dir @ parent_dir)
    child_dir /= np.linalg.norm(child_dir)
    absorbed_child_dir = child_dir  # child carries the child/parent semantics while parent latent stays off
    parent_only = latent["absorbed_parent_latent"]
    child = latent["child_latent"]
    activations = parent_only[:, None] * parent_dir[None, :] + child[:, None] * absorbed_child_dir[None, :]
    activations += noise * rng.normal(size=activations.shape)
    return HierarchyDataset(
        activations=activations,
        parent_label=latent["parent_label"],
        child_label=latent["child_label"],
        parent_direction=parent_dir,
        child_direction=child_dir,
        absorbed_parent_direction=absorbed_child_dir,
    )


def compare_flat_vs_hierarchical_codes(dataset: HierarchyDataset) -> dict[str, float]:
    """Compare parent recovery from a flat direction vs explicit hierarchy.

    The flat proxy computes parent activation by projecting onto parent direction.
    The hierarchical proxy treats child as implying parent.
    """
    flat_parent = np.maximum(0.0, dataset.activations @ dataset.parent_direction)
    child_proxy = np.maximum(0.0, dataset.activations @ dataset.absorbed_parent_direction)
    absorbed = absorption_score(flat_parent, child_proxy, dataset.parent_label)
    flat_pred = flat_parent > 0.5
    hierarchical_pred = np.logical_or(flat_parent > 0.5, child_proxy > 0.5)
    flat_error = np.mean(flat_pred != dataset.parent_label.astype(bool))
    hierarchical_error = np.mean(hierarchical_pred != dataset.parent_label.astype(bool))
    return {
        "absorption_score": float(absorbed),
        "flat_parent_error": float(flat_error),
        "hierarchical_parent_error": float(hierarchical_error),
    }
