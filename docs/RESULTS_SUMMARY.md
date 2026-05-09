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
Higher-order triple-gate contrast: `1.0000` using `8` intervention states.
Noisy recovery all-correct rate: `1.0000` at noise `0.005` and `0.0000` at noise `0.3`.
Noisy recovery abstention rate: `0.0000` at noise `0.005` and `1.0000` at noise `0.3`.
Active design mean measurements: `22.0000` at noise `0.0` and `79.9900` at noise `0.04` with accuracy `0.9113`.
Active-vs-baseline at noise `0.02`, budget `64`: active stable rate `1.0000`, uniform `0.9000`, random `0.0900`.

## Atlas Gluing

Best global glue error: `0.2500` versus context-indexed glue error `0.0000`.

## Certificate Vectors

Pareto frontier indices: `[0, 1]`.
Accepted certificates: `['compact_but_weak', 'faithful_but_long']`.

## Transition Atoms

Best MOLT/global MSE ratio: `0.0001` at gate separation `4.0` and noise `0.0`.

## Training Emergence

Mean mechanism lead: `8.6000` steps.
Mean mechanism/behavior correlation: `0.9574`.

