"""Ground-truth and synthetic benchmarks for MNF research."""

from mnf.benchmarks.synthetic import make_chain_program, sample_chain_data
from mnf.benchmarks.superposition import SparseFeatureWorld, superposition_phase_statistic
from mnf.benchmarks.hierarchy import make_hierarchy_dataset, compare_flat_vs_hierarchical_codes
from mnf.benchmarks.cyclic import make_weekday_rotation_dataset, weekday_space
from mnf.benchmarks.random_control import random_vs_trained_control

__all__ = [
    "make_chain_program",
    "sample_chain_data",
    "SparseFeatureWorld",
    "superposition_phase_statistic",
    "make_hierarchy_dataset",
    "compare_flat_vs_hierarchical_codes",
    "make_weekday_rotation_dataset",
    "weekday_space",
    "random_vs_trained_control",
]
