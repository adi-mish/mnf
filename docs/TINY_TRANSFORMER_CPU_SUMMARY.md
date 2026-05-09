# Tiny transformer CPU summary

This file summarizes `docs/tiny_transformer_cpu_sweep.json`, generated with:

```bash
python3 scripts/run_tiny_transformer_sweep.py --n-seeds 20 --steps 160 > docs/tiny_transformer_cpu_sweep.json
```

The benchmark trains a one-layer CPU transformer on modular addition over
`C_7`, then checks whether counterfactually shifting either input token rotates
the predicted output by the same modular offset.

## Result

- Seeds: `20`
- Training steps per seed: `160`
- Mean final accuracy: `0.9990`
- Minimum final accuracy: `0.9796`
- Mean input-a cyclic-shift consistency: `0.9980`
- Mean input-b cyclic-shift consistency: `0.9980`

## Interpretation

This is a learned-weight smoke test, not a large-language-model result. It
does show that the typed cyclic intervention machinery can be applied after
training a tiny transformer on CPU, closing one of the PLAN3 gaps between
purely executable synthetic programs and real learned weights.

The next step that changes the claim qualitatively is activation-local
mechanism recovery in external transformer tooling. Tracr, TransformerLens, and
Hugging Face `transformers` are not installed in this environment, and larger
model intervention studies are the point where GPU support becomes material.
