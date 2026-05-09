"""Small optional model components used by CPU smoke benchmarks."""

from mnf.models.tiny_transformer import (
    TinyModularAdditionTransformer,
    evaluate_activation_site_patching,
    evaluate_embedding_patch_interventions,
    evaluate_modular_interventions,
    modular_addition_dataset,
    torch_available,
    train_tiny_modular_addition,
)
from mnf.models.tiny_redundant import (
    TinyRedundantModularTransformer,
    evaluate_redundant_route_interventions,
    run_redundant_cpu_sweep,
    train_tiny_redundant_modular_transformer,
)

__all__ = [
    "TinyModularAdditionTransformer",
    "TinyRedundantModularTransformer",
    "torch_available",
    "modular_addition_dataset",
    "train_tiny_modular_addition",
    "train_tiny_redundant_modular_transformer",
    "evaluate_modular_interventions",
    "evaluate_embedding_patch_interventions",
    "evaluate_activation_site_patching",
    "evaluate_redundant_route_interventions",
    "run_redundant_cpu_sweep",
]
