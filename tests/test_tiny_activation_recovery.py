import pytest

from mnf.experiments.run_tiny_activation_recovery import run
from mnf.models import torch_available


@pytest.mark.slow
def test_tiny_activation_recovery_smoke():
    if not torch_available():
        return

    out = run(seeds=(0,), steps=80)

    assert out["available"] is True
    assert "encoder.layers.0.self_attn.pattern" in out["hook_sites"]
    assert out["meda_redundancy_recovered_rate"] == 1.0
    assert out["single_ablation_underweights_rate"] == 1.0
    assert out["acdc_redundancy_failure"]["misses_redundancy"] is True
