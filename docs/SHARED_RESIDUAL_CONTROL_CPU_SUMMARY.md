# Shared-residual decorative-route control CPU summary

This file summarizes `docs/shared_residual_control_cpu_sweep.json`, generated
with:

```bash
python3 scripts/run_tiny_shared_residual_control_sweep.py --n-seeds 100 --steps 160 > docs/shared_residual_control_cpu_sweep.json
```

The benchmark trains the same shared encoder/residual-stream architecture as
the shared-residual redundant transformer, but freezes the second readout route
at zero. The live route must solve modular addition over `C_7`; the decorative
route should remain at chance. A valid redundancy certificate must therefore
reject the pair.

## Result

- Seeds: `100`
- Training steps per seed: `160`
- Mean base accuracy: `0.9984`
- Mean live-route-only accuracy: `0.9984`
- Mean decorative-route-only accuracy: `0.1429`
- Mean both-routes-ablated accuracy: `0.1429`
- Mean max single-ablation drop: `0.8555`
- Mean dual-ablation drop: `0.8555`
- Redundancy certified rate: `0.0000`
- Single-ablation underweights rate: `0.0000`
- False redundancy rate: `0.0000`

## Interpretation

This is a negative control for the shared-MDL route-reuse claim. It shows that
the current certificate does not mark a route as redundant merely because it is
named, shares the residual stream, or exists in a two-head architecture. The
positive shared-residual result therefore has a useful paired falsification
test: both readouts must be independently sufficient, not merely present.
