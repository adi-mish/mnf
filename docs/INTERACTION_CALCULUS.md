# Interaction calculus

This document is the executable iMNF interaction layer in one place. It explains
what the current CPU-only implementation can measure before real-model hooks or
GPU experiments are needed.

## Pairwise factorial table

For mechanisms `m` and `n`, define:

```text
Y_ab = E[B | m = a, n = b]
```

where `a,b in {0,1}` and larger `B` means stronger behavior. The implementation
is `mnf/interactions/factorial_effects.py`.

Core contrasts:

```text
joint_effect = Y_11 - Y_00
synergy = Y_11 - Y_10 - Y_01 + Y_00
gate_m_to_n = (Y_11 - Y_10) - (Y_01 - Y_00)
gate_n_to_m = (Y_11 - Y_01) - (Y_10 - Y_00)
```

Redundancy is detected when both single ablations look unimportant but the dual
ablation matters:

```text
Y_11 ~= Y_10 ~= Y_01
Y_00 << Y_11
```

This is the formal failure mode for isolated-circuit tests.

## Intervention design

Full factorial recovery costs:

```text
2^K
```

states for `K` mechanisms. The pairwise all-on-context design costs:

```text
1 + K + K(K - 1) / 2
```

states after deduplication: one all-on baseline, `K` single ablations, and every
dual ablation. This is implemented in `mnf/interactions/design.py`.

This design is not a substitute for high-order discovery. It is a first pass
that estimates conditional pairwise redundancy, gating, additive effects, and
competition cheaply in the selected context.

For a selected subset `S`, the implementation also supports sparse higher-order
inclusion-exclusion:

```text
Delta_S =
  sum_{T subseteq S} (-1)^(|S| - |T|) Y_T
```

where mechanisms in `T` are on, mechanisms in `S \\ T` are off, and all
non-subset mechanisms are held in the selected context. The current
triple-gate benchmark has third-order contrast `1.0`.

## Uncertainty

`mnf/interactions/uncertainty.py` bootstraps confidence intervals for derived
quantities such as synergy, redundancy score, compensation score, and gate
strength. In the current generated results:

```text
gate-synergy CI lower bound ~= 0.989
redundancy-score CI lower bound ~= 0.993
```

on the noisy synthetic recovery suite.

`mnf/interactions/bounds.py` also gives deterministic worst-case bounds. If
each factorial cell mean has absolute error at most `epsilon`, then a linear
contrast `sum_i c_i Y_i` has error at most `epsilon * sum_i |c_i|`. Pairwise
synergy and gate contrasts therefore have error at most `4 epsilon`, and an
order-`k` inclusion-exclusion contrast has error at most `2^k epsilon`.

The classifier now has an abstaining variant:
`classify_pairwise_interaction_with_uncertainty`. It returns `uncertain` when
the label chosen at the nominal tolerance changes after modestly widening the
error bar. This keeps weak noisy contrasts out of hard additive/non-additive
claims.

## Recovery metrics

`mnf/interactions/recovery.py` provides:

- thresholded edge recovery;
- pair-label recovery;
- precision, recall, F1, and label accuracy.

The current recovery suite has six mechanisms and fifteen mechanism pairs:

```text
redundant_left, redundant_right
gate, worker
competitor
additive
```

It recovers:

```text
1 redundant pair
1 synergistic/gated pair
1 competitive pair
12 additive-context pairs
```

from `22` intervention states with F1 `1.0`.

The noisy recovery sweep records how this degrades as cell observations are
perturbed. It also reports the abstention rate from the margin-stability
classifier. This is a synthetic margin check, not a substitute for real
activation intervention uncertainty.

## Context Stability

Pairwise labels are selected-context claims. Holding all non-pair mechanisms on
can hide interactions that appear when the background context changes.
`mnf/benchmarks/interactions/context_stability.py` is a minimal counterexample:
the `left/right` pair is additive when `switch` is on and synergistic when
`switch` is off. The correct output is therefore not one universal pair label,
but a context-stability report.

## Active Design

`mnf/interactions/active.py` implements a small active-design loop. It starts
from the pairwise all-on-context candidate design, samples the intervention
states needed by unresolved pair labels, then allocates repeated measurements
to states that participate in unstable contrasts.

In the exact six-mechanism recovery suite, the active loop terminates after the
same `22` unique states required by the deduplicated pairwise design. Under
noise, it does not invent new states; it spends additional measurements on the
same state set until contrast margins clear the deterministic tolerance or the
budget is exhausted.

This is not an optimal experimental-design theorem. It is a CPU-checkable
margin rule for deciding where repeated intervention measurements are useful.
The benchmark also reports matched-budget uniform and random repeated designs,
so the active rule is compared against simple non-adaptive alternatives rather
than left as an unbaselined heuristic.

## What remains before real models

The CPU implementation now covers the pairwise interaction engine, shared-MDL
accounting, uncertainty, active repeated-measurement design, and synthetic
recovery. The next CPU-only gaps are:

- higher-order interaction search over many candidate subsets;
- richer context-shifted environments beyond the current minimal counterexample;
- proof hardening for ecosystem-level approximate/noisy settings.

The next non-CPU-local gap is real-model activation intervention: TransformerLens
or equivalent hooks, model weights, and likely GPU for anything beyond tiny
models.
