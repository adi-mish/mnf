# Shared-residual tiny transformer CPU summary

This file summarizes `docs/shared_residual_tiny_transformer_cpu_sweep.json`,
generated with:

```bash
python3 scripts/run_tiny_shared_residual_transformer_sweep.py --n-seeds 100 --steps 160 > docs/shared_residual_tiny_transformer_cpu_sweep.json
```

The benchmark trains one shared encoder/residual stream with two redundant
readout routes for modular addition over `C_7`. Each route must solve the task
alone, but both routes reuse the same learned residual state.

## Result

- Seeds: `100`
- Training steps per seed: `160`
- Mean base accuracy: `0.9994`
- Mean route-a-only accuracy: `0.9994`
- Mean route-b-only accuracy: `0.9994`
- Mean both-routes-ablated accuracy: `0.1429`
- Mean max single-ablation drop: `0.0000`
- Mean dual-ablation drop: `0.8565`
- Redundancy certified rate: `1.0000`
- Single-ablation underweights rate: `1.0000`
- Shared residual parameter gain: `8896` parameters, or `0.4873` of the
  independent-route parameter count.

## Interpretation

This is stronger than the two-stack redundant transformer result because the
routes reuse one shared residual stream. It still has named readout routes, so
it is not hidden-route discovery inside an arbitrary pretrained transformer.
It does show that the learned shared state supports two independently sufficient
mechanistic readouts and that shared-MDL accounting correctly rewards reuse of
the common encoder.
