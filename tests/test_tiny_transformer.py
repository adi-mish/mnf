import pytest

from mnf.models import evaluate_modular_interventions, torch_available, train_tiny_modular_addition


@pytest.mark.slow
def test_tiny_transformer_learns_modular_addition_on_cpu():
    if not torch_available():
        return

    model, training = train_tiny_modular_addition(seed=0, steps=120)
    interventions = evaluate_modular_interventions(model)

    assert training.final_accuracy >= 0.95
    assert interventions["base_accuracy"] >= 0.95
    assert interventions["a_cyclic_shift_consistency"] >= 0.95
    assert interventions["b_cyclic_shift_consistency"] >= 0.95
