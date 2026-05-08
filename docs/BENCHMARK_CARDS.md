# MechanismLab benchmark cards

These cards document the current CPU-only synthetic benchmark suite. Each
benchmark is designed to distinguish labelability from intervention-predictive
mechanism recovery.

## Scalar chain

- **File:** `mnf/benchmarks/synthetic.py`
- **Mechanism:** `x -> a -> b -> y`
- **State types:** scalar
- **Interventions:** clamp hidden scalar `a`; path-block candidate edges
- **Expected failure mode:** correlation discovery over-proposes transitive edges
- **Current result:** intervention pruning recovers direct edges with graph F1 `1.0`

## Gated XOR

- **File:** `mnf/benchmarks/gated.py`
- **Mechanism:** `x1, x2 -> xor`; `gate, xor -> y`
- **State types:** binary
- **Interventions:** flip or clamp the gate and input bits
- **Expected failure mode:** linear state-only methods miss nonlinear parity
- **Current result:** intervention pruning recovers the four direct edges with graph F1 `1.0`

## Modular addition

- **File:** `mnf/benchmarks/modular.py`
- **Mechanism:** `(a + b) mod 7`
- **State types:** cyclic `C_7`
- **Interventions:** shift one addend and check corresponding output rotation
- **Expected failure mode:** scalar encodings fragment a cyclic variable
- **Current result:** modular shift error `0.0`

## Relational lookup

- **File:** `mnf/benchmarks/relational.py`
- **Mechanism:** `(subject, relation) -> object`
- **State types:** categorical subject, relation, and object
- **Interventions:** counterfactual relation swaps
- **Expected failure mode:** probes can memorize object labels without recovering key-value structure
- **Current result:** relation swap maps city queries to instrument objects exactly

## Hierarchical absorption

- **File:** `mnf/benchmarks/hierarchy.py`
- **Mechanism:** child implies parent
- **State types:** binary parent and child features with hierarchy
- **Interventions:** child-implies-parent consistency check
- **Expected failure mode:** flat sparse dictionaries absorb parent mass into child features
- **Current result:** phase sweep average flat parent error `0.2247`; hierarchical error `0.0`

## Cyclic weekday

- **File:** `mnf/benchmarks/cyclic.py`
- **Mechanism:** weekday state on a learned 2D circular plane
- **State types:** cyclic `C_7`
- **Interventions:** rotate weekday by fixed steps in the chart
- **Expected failure mode:** direction-only charts lack a natural rotation operation
- **Current result:** rotation error stays `0.0` through noise `0.08`; at noise `0.3`, error is about `0.147`

## Sparse superposition

- **File:** `mnf/benchmarks/superposition.py`
- **Mechanism:** sparse overcomplete feature dictionary
- **State types:** sparse scalar latent features
- **Interventions:** dictionary-level latent perturbations
- **Expected failure mode:** neuron basis is not aligned to learned feature dictionary
- **Current result:** superposition favored in `33/48` grid conditions in the current sweep

## Transition-only map

- **File:** `mnf/benchmarks/transition_only.py`
- **Mechanism:** gated piecewise-linear transform
- **State types:** transition atoms with a gate
- **Interventions:** compare global linear replacement against MOLT-like gated transforms
- **Expected failure mode:** state-only or global-linear explanations miss conditional transformations
- **Current result:** at gate separation `2.5` and noise `0.05`, MOLT/global MSE ratio is about `0.021`

## Random vs trained control

- **File:** `mnf/benchmarks/random_control.py`
- **Mechanism:** labelable random representation versus label-routed trained-style representation
- **State types:** binary label direction
- **Interventions:** measure whether the decoded feature predicts output changes
- **Expected failure mode:** labelability is mistaken for causal use
- **Current result:** both systems have labelability `0.961`; causal-use scores are `0.012` random versus `0.900` trained-style

## Shortcut control

- **File:** `mnf/benchmarks/shortcut.py`
- **Mechanism:** causal feature remains stable; shortcut feature shifts across environments
- **State types:** binary causal and shortcut directions
- **Interventions:** cross-environment invariance filter
- **Expected failure mode:** train-only probes select the shortcut
- **Current result:** shortcut train accuracy `0.979`, shifted accuracy `0.105`; causal feature is accepted and shortcut is rejected

## Shortcut selector comparison

- **File:** `mnf/baselines/selection.py`
- **Mechanism:** hard shortcut setting where the shortcut has higher train accuracy than the weaker causal feature
- **State types:** binary causal and shortcut directions
- **Interventions:** compare train-only selection against invariance-aware selection
- **Expected failure mode:** labelability baseline selects the train shortcut
- **Current result:** train-only selector chooses `shortcut` and gets shifted accuracy below `0.2`; invariance selector chooses `causal` and gets shifted accuracy above `0.9`

## Memorizing alignment

- **File:** `mnf/benchmarks/memorization.py`
- **Mechanism:** arbitrary lookup from sample IDs to labels
- **State types:** high-cardinality sample identity
- **Interventions:** held-out identity test and description-length penalty
- **Expected failure mode:** unconstrained alignment gets perfect observation fit by memorization
- **Current result:** train accuracy `1.0`, test accuracy `0.475`, description length `512.0`
