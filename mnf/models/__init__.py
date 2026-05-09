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
from mnf.models.tiny_shared_residual import (
    TinySharedResidualRedundantTransformer,
    evaluate_shared_residual_route_interventions,
    run_shared_residual_redundant_cpu_sweep,
    run_shared_residual_single_route_control_sweep,
    train_tiny_shared_residual_redundant_transformer,
    train_tiny_shared_residual_single_route_control,
)

__all__ = [
    "TinyModularAdditionTransformer",
    "TinyRedundantModularTransformer",
    "TinySharedResidualRedundantTransformer",
    "torch_available",
    "modular_addition_dataset",
    "train_tiny_modular_addition",
    "train_tiny_redundant_modular_transformer",
    "train_tiny_shared_residual_redundant_transformer",
    "train_tiny_shared_residual_single_route_control",
    "evaluate_modular_interventions",
    "evaluate_embedding_patch_interventions",
    "evaluate_activation_site_patching",
    "evaluate_redundant_route_interventions",
    "evaluate_shared_residual_route_interventions",
    "run_redundant_cpu_sweep",
    "run_shared_residual_redundant_cpu_sweep",
    "run_shared_residual_single_route_control_sweep",
]
