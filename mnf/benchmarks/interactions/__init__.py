"""Mechanism ecology benchmarks for Interactive MNF."""

from mnf.benchmarks.interactions.capacity_competition import capacity_competition_score, capacity_competition_sweep
from mnf.benchmarks.interactions.cooperative_routing import cooperative_routing_metrics
from mnf.benchmarks.interactions.developmental_bootstrap import developmental_bootstrap_metrics
from mnf.benchmarks.interactions.gating_mechanism import gated_worker_behavior, gating_metrics
from mnf.benchmarks.interactions.mechanism_death import mechanism_death_metrics
from mnf.benchmarks.interactions.phase_diagram import factorial_interaction_phase_diagram, mixed_or_and_behavior
from mnf.benchmarks.interactions.redundant_paths import redundant_or_behavior, redundant_paths_metrics
from mnf.benchmarks.interactions.shared_atom_reuse import make_shared_atom_ecosystem, shared_atom_reuse_metrics
from mnf.benchmarks.interactions.synergistic_paths import and_synergy_behavior, synergistic_paths_metrics

__all__ = [
    "capacity_competition_score",
    "capacity_competition_sweep",
    "cooperative_routing_metrics",
    "developmental_bootstrap_metrics",
    "gated_worker_behavior",
    "gating_metrics",
    "mechanism_death_metrics",
    "factorial_interaction_phase_diagram",
    "mixed_or_and_behavior",
    "redundant_or_behavior",
    "redundant_paths_metrics",
    "make_shared_atom_ecosystem",
    "shared_atom_reuse_metrics",
    "and_synergy_behavior",
    "synergistic_paths_metrics",
]
