"""Small optional model components used by CPU smoke benchmarks."""

from mnf.models.tiny_transformer import (
    TinyModularAdditionTransformer,
    evaluate_embedding_patch_interventions,
    evaluate_modular_interventions,
    modular_addition_dataset,
    torch_available,
    train_tiny_modular_addition,
)

__all__ = [
    "TinyModularAdditionTransformer",
    "torch_available",
    "modular_addition_dataset",
    "train_tiny_modular_addition",
    "evaluate_modular_interventions",
    "evaluate_embedding_patch_interventions",
]
