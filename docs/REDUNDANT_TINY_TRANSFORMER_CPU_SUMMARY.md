# Redundant tiny transformer CPU summary

This file summarizes `docs/redundant_tiny_transformer_cpu_sweep.json`,
generated with:

```bash
python3 scripts/run_tiny_redundant_transformer_sweep.py --n-seeds 100 --steps 160 > docs/redundant_tiny_transformer_cpu_sweep.json
```

The benchmark trains a two-route CPU transformer on modular addition over
`C_7`. The training loss requires the full model, route-a alone, and route-b
alone to solve the task. Evaluation then compares single-route ablations against
dual-route ablation.

## Result

- Seeds: `100`
- Training steps per seed: `160`
- Mean base accuracy: `1.0000`
- Mean route-a-only accuracy: `0.9986`
- Mean route-b-only accuracy: `0.9949`
- Mean both-routes-ablated accuracy: `0.1429`
- Mean max single-ablation drop: `0.0061`
- Mean dual-ablation drop: `0.8571`
- Redundancy certified rate: `0.9900`
- Single-ablation underweights rate: `0.9900`

## Interpretation

This is the strongest PLAN4 CPU result so far. It is a learned-weight
counterexample to isolated single-ablation necessity: in almost every seed,
ablating either route alone barely changes behavior, while ablating both routes
collapses accuracy to chance. The result is intentionally architecture-explicit:
the two routes are separately parameterized, so it proves the recovery and
certificate logic on learned weights rather than proving that hidden redundant
routes can always be discovered inside arbitrary pretrained transformers.
