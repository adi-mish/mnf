# Tiny transformer CPU summary

This file summarizes `docs/tiny_transformer_cpu_sweep.json`, generated with:

```bash
python3 scripts/run_tiny_transformer_sweep.py --n-seeds 100 --steps 160 > docs/tiny_transformer_cpu_sweep.json
```

The benchmark trains a one-layer CPU transformer on modular addition over
`C_7`, then checks whether counterfactually shifting either input token rotates
the predicted output by the same modular offset. PLAN4 adds activation-local
checks: source-token embedding outputs are patched into the base run at one
input position, and later block activations at the final readout token are
patched from source runs.

## Result

- Seeds: `100`
- Training steps per seed: `160`
- Mean final accuracy: `0.9986`
- Minimum final accuracy: `0.9592`
- Mean input-a cyclic-shift consistency: `0.9971`
- Mean input-b cyclic-shift consistency: `0.9971`
- Mean input-a embedding-patch consistency: `0.9971`
- Mean input-b embedding-patch consistency: `0.9971`
- Mean deeper activation-site patch consistency:
  - `embedding`: `0.9971`
  - `encoder.layers.0.norm1`: `0.9971`
  - `encoder.layers.0.norm2`: `0.9971`
- Mean wrong-token patch consistency controls:
  - `embedding`: `0.0000`
  - `encoder.layers.0.norm1`: `0.0000`
  - `encoder.layers.0.norm2`: `0.0000`
- Minimum embedding-patch consistency across seeds and input positions:
  `0.9184`

## Interpretation

This is a learned-weight smoke test, not a large-language-model result. It
does show that the typed cyclic intervention machinery can be applied after
training a tiny transformer on CPU, and that the same counterfactual claim can
be checked by patching learned activations at the embedding-output site and at
later residual-stream normalization sites. Wrong-token controls stay at `0.0`,
so the positive effect is not a generic consequence of injecting any shifted
source activation. This closes one PLAN4 CPU gap, but only in a tiny model with
a simple modular task.

The next step that changes the claim qualitatively is activation-local
mechanism recovery in external transformer tooling. Tracr, TransformerLens, and
Hugging Face `transformers` are not installed in this environment, and larger
model intervention studies are the point where GPU support becomes material.
