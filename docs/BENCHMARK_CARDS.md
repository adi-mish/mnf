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

## Cyclic scalar baseline

- **File:** `mnf/baselines/cyclic.py`
- **Mechanism:** weekday state on a circle, compared to scalar integer regression
- **State types:** typed cyclic `C_7` versus scalar label chart
- **Interventions:** rotate in the typed chart versus add to the scalar label
- **Expected failure mode:** scalar regression breaks at the cyclic wraparound
- **Current result:** typed chart has near-zero low-noise error; scalar baseline has label/rotation error above `0.4`

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

## Training emergence

- **File:** `mnf/benchmarks/training_dynamics.py`
- **Mechanism:** smooth mechanism formation with thresholded behavioral metric
- **State types:** scalar mechanism-progress trace
- **Interventions:** compare intervention-progress crossing time against behavioral crossing time
- **Expected failure mode:** behavior-only analysis misses gradual mechanism formation
- **Current result:** mechanism progress crosses threshold before behavioral emergence in the synthetic trace

## Induction match-copy

- **File:** `mnf/benchmarks/induction.py`
- **Mechanism:** find the previous occurrence of the final bigram and copy the following token
- **State types:** categorical tokens and match position
- **Interventions:** held-out pair generalization and match-position state inspection
- **Expected failure mode:** pair memorization fits train pairs but fails on unseen pairs
- **Current result:** algorithmic copy mechanism gets held-out accuracy `1.0`; pair memorizer is near chance on held-out pairs

## Redundant paths

- **File:** `mnf/benchmarks/interactions/redundant_paths.py`
- **Mechanism:** either of two routes is sufficient for the behavior
- **State types:** binary mechanism-strength states
- **Interventions:** pairwise factorial on/off ablations
- **Expected failure mode:** single ablation incorrectly rejects both real mechanisms
- **Current result:** single-ablation drops are `0.0`, but dual-ablation drop is `1.0`

## Gating interaction

- **File:** `mnf/benchmarks/interactions/gating_mechanism.py`
- **Mechanism:** a gate enables a downstream worker mechanism
- **State types:** binary gate and worker states
- **Interventions:** conditional worker ablation under gate-on and gate-off states
- **Expected failure mode:** marginal interventions miss a mechanism that acts by enabling another mechanism
- **Current result:** worker effect is `1.0` when the gate is on and `0.0` when the gate is off

## Shared atom reuse

- **File:** `mnf/benchmarks/interactions/shared_atom_reuse.py`
- **Mechanism:** two lookup mechanisms reuse a route atom
- **State types:** route atom plus mechanism-specific state atoms
- **Interventions:** shared-versus-independent description-length accounting
- **Expected failure mode:** independent circuit descriptions duplicate the same reusable variable
- **Current result:** shared-MDL gain is positive in the route-reuse toy ecosystem

## Capacity competition

- **File:** `mnf/benchmarks/interactions/capacity_competition.py`
- **Mechanism:** two mechanisms coactivate on aligned decoder directions
- **State types:** fuzzy memberships over sparse atoms
- **Interventions:** sweep coactivation, decoder alignment, and atom overlap
- **Expected failure mode:** treating mechanisms independently misses representational interference
- **Current result:** competition increases monotonically with coactivation and decoder alignment

## Interaction phase diagram

- **File:** `mnf/benchmarks/interactions/phase_diagram.py`
- **Mechanism:** tunable mixture of redundant OR-style and synergistic AND-style pathways
- **State types:** binary mechanism-strength states
- **Interventions:** pairwise factorial on/off ablations over a grid of mixture weights
- **Expected failure mode:** a binary mechanism/not-mechanism view misses continuous transitions between redundancy and synergy
- **Current result:** the sweep spans redundant and synergistic/gated regimes

## Developmental bootstrap

- **File:** `mnf/benchmarks/interactions/developmental_bootstrap.py`
- **Mechanism:** one mechanism becomes learnable only after a supporting mechanism grows
- **State types:** scalar mechanism-strength traces
- **Interventions:** coupled mechanism ecology dynamics
- **Expected failure mode:** behavior-only analysis misses support-driven delayed emergence
- **Current result:** the dependent mechanism crosses threshold after the scaffold mechanism

## Mechanism death

- **File:** `mnf/benchmarks/interactions/mechanism_death.py`
- **Mechanism:** a stronger mechanism suppresses a competing mechanism through the ecology dynamics
- **State types:** scalar mechanism-strength traces
- **Interventions:** coupled competition dynamics
- **Expected failure mode:** static circuit snapshots miss training-time suppression
- **Current result:** the suppressed mechanism loses strength as the winner grows
