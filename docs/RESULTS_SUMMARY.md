# Research sweep summary

This file is generated from `docs/research_sweeps.json`.

## Ground Truth Recovery

| Benchmark | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| scalar_chain | 1.0000 | 1.0000 | 1.0000 |
| gated_xor | 1.0000 | 1.0000 | 1.0000 |
| modular_addition_C7 | 1.0000 | 1.0000 | 1.0000 |
| relational_lookup | 1.0000 | 1.0000 | 1.0000 |

## Shortcut Baseline

| Selector | Selected | Train Accuracy | Shifted Accuracy | Invariance Gap |
| --- | --- | ---: | ---: | ---: |
| train-only | shortcut | 0.9805 | 0.1075 | 0.8730 |
| invariance | causal | 0.9620 | 0.9635 | 0.0015 |

## Feature Baselines

Mean false-mechanism rate on random labelable controls: `0.6667`.
Mean causal-use score: random labelable `0.0250` versus trained-used `0.8285`.
Dictionary baselines false-mechanism rate on random labelable controls: `1.0000`.
ACDC-style single-ablation redundancy control misses redundancy: `True` with dual drop `1.0000`.

## Cyclic Baseline

At noise `0.0`, typed rotation error is `0.0000` versus scalar rotation error `0.5729`.
At noise `0.3`, typed rotation error is `0.1516` versus scalar rotation error `0.6218`.

## Induction Match-Copy

Mean held-out memorizer accuracy: `0.0544`.
Mean held-out copy-mechanism accuracy: `1.0000`.

## Mechanism Interactions

Redundant-path dual ablation drop: `1.0000` with single-ablation drops `0.0000` and `0.0000`.
Gating strength `gate -> worker`: `1.0000`.
Shared-MDL gain for reused route atom: `1.7500`.
Maximum capacity-competition score in the sweep: `2.8000`.
Factorial interaction phase-diagram rows: `49`.
Interaction recovery F1: `1.0000` over `15` mechanism pairs using `22` intervention states.
Structural aliasing: behavior-only label `ambiguous_without_internal_evidence`; internal evidence orients the directed gate as `m_to_n`.
Context-stability changed pairs: `3` of `6`; the focal pair changes from `additive` to `synergistic_or_gated`.
Context-stability sweep changed-context rate: `0.7500` over `16` parameter settings.
Higher-order triple-gate contrast: `1.0000` using `8` intervention states.
Pairwise-only higher-order search labels: `{'unobserved': 1}`; order-3 search labels: `{'positive_higher_order': 1}`.
Noisy recovery all-correct rate: `1.0000` at noise `0.005` and `0.0000` at noise `0.3`.
Noisy recovery abstention rate: `0.0000` at noise `0.005` and `1.0000` at noise `0.3`.
Active design mean measurements: `22.0000` at noise `0.0` and `79.9900` at noise `0.04` with accuracy `0.9113`.
Active-vs-baseline at noise `0.02`, budget `64`: active stable rate `1.0000`, uniform `0.9000`, random `0.0900`.

## Atlas Gluing

Best global glue error: `0.2500` versus context-indexed glue error `0.0000`.

## Certificate Vectors

Pareto frontier indices: `[0, 1]`.
Accepted certificates: `['compact_but_weak', 'faithful_but_long']`.
Example identification set: `['faithful_route_a', 'faithful_route_b']` with diameter `0.0300`.

## Identifiability

Output-only atom-splitting max distance: `0.0000` with representatives `['split_routes', 'merged_route']` and ID diameter `1.0000`.
Adding internal route markers raises max distance to `1.0000`.

## Transition Atoms

Best MOLT/global MSE ratio: `0.0001` at gate separation `4.0` and noise `0.0`.

## Tiny Transformer

Mean final modular-addition accuracy: `1.0000`.
Mean cyclic-shift consistency: input-a `1.0000`, input-b `1.0000`.
Mean embedding-patch consistency: input-a `1.0000`, input-b `1.0000`.
Mean deeper activation-site patch consistency: embedding: `1.0000`, encoder.layers.0.norm1: `1.0000`, encoder.layers.0.norm2: `1.0000`.
Mean wrong-token patch consistency controls: embedding: `0.0000`, encoder.layers.0.norm1: `0.0000`, encoder.layers.0.norm2: `0.0000`.

## Tiny Redundant Transformer

Mean base accuracy: `1.0000`.
Mean route-only accuracy: route-a `0.9932`, route-b `0.9796`.
Mean both-routes-ablated accuracy: `0.1429`.
Mean max single-ablation drop: `0.0204` versus dual-ablation drop `0.8571`.
Redundancy certified rate: `1.0000`; single-ablation underweights rate: `1.0000`.

## Tiny Shared-Residual Redundant Transformer

Mean base accuracy: `1.0000`.
Mean route-only accuracy: route-a `1.0000`, route-b `1.0000`.
Mean both-routes-ablated accuracy: `0.1429`.
Mean max single-ablation drop: `0.0000` versus dual-ablation drop `0.8571`.
Redundancy certified rate: `1.0000`; single-ablation underweights rate: `1.0000`.
Shared residual parameter gain: `8896` parameters (`0.4873` of independent-route parameters).

## Tiny Shared-Residual Decorative-Route Control

Mean base accuracy: `1.0000`.
Mean route-only accuracy: live route `1.0000`, decorative route `0.1429`.
Mean max single-ablation drop: `0.8571`.
False redundancy rate: `0.0000`.

## Tiny Activation Recovery

Hook sites captured: `['embedding', 'encoder.layers.0.linear1', 'encoder.layers.0.linear2', 'encoder.layers.0.norm1', 'encoder.layers.0.norm2', 'encoder.layers.0.self_attn.k', 'encoder.layers.0.self_attn.out', 'encoder.layers.0.self_attn.pattern', 'encoder.layers.0.self_attn.q', 'encoder.layers.0.self_attn.v', 'output', 'position']`.
MEDA redundancy recovery rate: `1.0000`.
Single-ablation underweights rate: `1.0000`.

## Training Emergence

Mean mechanism lead: `8.6000` steps.
Mean mechanism/behavior correlation: `0.9574`.

## Real-Model CPU Smoke

Model `sshleifer/tiny-gpt2` produced logits shape `[1, 3, 50257]` with `3` hidden-state tensors.
TransformerLens random hooked-model smoke available: `True` with `23` cache entries.
Tracr compiled-program smoke available: `True` with all-exact-match `True` over `11` programs.

