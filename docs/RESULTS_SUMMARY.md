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

## Transition Atoms

Best MOLT/global MSE ratio: `0.0001` at gate separation `4.0` and noise `0.0`.

## Training Emergence

Mean mechanism lead: `8.6000` steps.
Mean mechanism/behavior correlation: `0.9574`.

