"""Ground-truth and synthetic benchmarks for MNF research."""

from mnf.benchmarks.synthetic import make_chain_program, sample_chain_data
from mnf.benchmarks.superposition import SparseFeatureWorld, superposition_phase_statistic
from mnf.benchmarks.hierarchy import make_hierarchy_dataset, compare_flat_vs_hierarchical_codes
from mnf.benchmarks.cyclic import make_weekday_rotation_dataset, weekday_space
from mnf.benchmarks.random_control import random_vs_trained_control
from mnf.benchmarks.gated import make_gated_xor_program, sample_gated_xor_data
from mnf.benchmarks.modular import make_modular_addition_program, sample_modular_addition_data
from mnf.benchmarks.relational import make_relational_lookup_program, sample_relational_lookup_data
from mnf.benchmarks.transition_only import make_transition_only_dataset, compare_global_linear_vs_molt
from mnf.benchmarks.shortcut import make_shortcut_dataset, shortcut_control
from mnf.benchmarks.memorization import memorization_control
from mnf.benchmarks.training_dynamics import make_training_emergence_trace, training_emergence_control
from mnf.benchmarks.induction import make_induction_program, compare_induction_vs_memorization

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
    "make_gated_xor_program",
    "sample_gated_xor_data",
    "make_modular_addition_program",
    "sample_modular_addition_data",
    "make_relational_lookup_program",
    "sample_relational_lookup_data",
    "make_transition_only_dataset",
    "compare_global_linear_vs_molt",
    "make_shortcut_dataset",
    "shortcut_control",
    "memorization_control",
    "make_training_emergence_trace",
    "training_emergence_control",
    "make_induction_program",
    "compare_induction_vs_memorization",
]
