import pytest

from mnf.models import (
    evaluate_activation_site_patching,
    evaluate_embedding_patch_interventions,
    evaluate_modular_interventions,
    evaluate_redundant_route_interventions,
    evaluate_shared_residual_route_interventions,
    tiny_activation_site_names,
    torch_available,
    train_tiny_modular_addition,
    train_tiny_redundant_modular_transformer,
    train_tiny_shared_residual_redundant_transformer,
    train_tiny_shared_residual_single_route_control,
)


@pytest.mark.slow
def test_tiny_transformer_learns_modular_addition_on_cpu():
    if not torch_available():
        return

    model, training = train_tiny_modular_addition(seed=0, steps=120)
    interventions = evaluate_modular_interventions(model)
    patching = evaluate_embedding_patch_interventions(model)
    site_patching = evaluate_activation_site_patching(model)

    assert training.final_accuracy >= 0.95
    assert interventions["base_accuracy"] >= 0.95
    assert interventions["a_cyclic_shift_consistency"] >= 0.95
    assert interventions["b_cyclic_shift_consistency"] >= 0.95
    assert patching["a_embedding_patch_consistency"] >= 0.95
    assert patching["b_embedding_patch_consistency"] >= 0.95
    assert site_patching["encoder.layers.0.norm1"]["a_patch_consistency"] >= 0.95
    assert site_patching["encoder.layers.0.norm1"]["b_patch_consistency"] >= 0.95
    assert site_patching["encoder.layers.0.norm2"]["a_patch_consistency"] >= 0.95
    assert site_patching["encoder.layers.0.norm2"]["b_patch_consistency"] >= 0.95
    assert site_patching["encoder.layers.0.norm1"]["a_wrong_token_consistency"] <= 0.2
    assert site_patching["encoder.layers.0.norm1"]["b_wrong_token_consistency"] <= 0.2
    assert site_patching["encoder.layers.0.norm2"]["a_wrong_token_consistency"] <= 0.2
    assert site_patching["encoder.layers.0.norm2"]["b_wrong_token_consistency"] <= 0.2
    assert "encoder.layers.0.self_attn" in tiny_activation_site_names(model)


@pytest.mark.slow
def test_tiny_redundant_transformer_recovers_redundant_routes_on_cpu():
    if not torch_available():
        return

    model, training = train_tiny_redundant_modular_transformer(seed=0, steps=120)
    interventions = evaluate_redundant_route_interventions(model)

    assert training.base_accuracy >= 0.95
    assert interventions["route_a_only_accuracy"] >= 0.95
    assert interventions["route_b_only_accuracy"] >= 0.95
    assert interventions["both_routes_ablated_accuracy"] <= 0.2
    assert interventions["max_single_ablation_drop"] <= 0.05
    assert interventions["dual_ablation_drop"] >= 0.75
    assert interventions["single_ablation_underweights"] is True
    assert interventions["redundancy_certified"] is True


@pytest.mark.slow
def test_tiny_shared_residual_transformer_recovers_redundant_readouts_on_cpu():
    if not torch_available():
        return

    model, training = train_tiny_shared_residual_redundant_transformer(seed=0, steps=120)
    interventions = evaluate_shared_residual_route_interventions(model)
    accounting = model.parameter_accounting()

    assert training.base_accuracy >= 0.95
    assert interventions["route_a_only_accuracy"] >= 0.95
    assert interventions["route_b_only_accuracy"] >= 0.95
    assert interventions["both_routes_ablated_accuracy"] <= 0.2
    assert interventions["max_single_ablation_drop"] <= 0.1
    assert interventions["dual_ablation_drop"] >= 0.75
    assert interventions["single_ablation_underweights"] is True
    assert interventions["redundancy_certified"] is True
    assert accounting["shared_parameter_gain"] > 0


@pytest.mark.slow
def test_tiny_shared_residual_control_rejects_decorative_route_on_cpu():
    if not torch_available():
        return

    model, training = train_tiny_shared_residual_single_route_control(seed=0, steps=120)
    interventions = evaluate_shared_residual_route_interventions(model)

    assert training.base_accuracy >= 0.95
    assert interventions["route_a_only_accuracy"] >= 0.95
    assert interventions["route_b_only_accuracy"] <= 0.2
    assert interventions["max_single_ablation_drop"] >= 0.75
    assert interventions["redundancy_certified"] is False
    assert interventions["single_ablation_underweights"] is False
