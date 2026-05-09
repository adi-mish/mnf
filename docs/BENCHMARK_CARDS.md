# MechanismLab benchmark cards

These cards document the current CPU-only synthetic benchmark suite. Each
benchmark is designed to distinguish labelability from intervention-predictive
mechanism recovery.

PLAN4 adds a stricter metadata requirement: benchmark cards should expose known
non-identifiabilities instead of implying that every ground-truth factor is
point-identified by the available observations.

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

## Feature baseline suite

- **File:** `mnf/baselines/features.py`
- **Mechanism:** labelable representation is compared with a representation whose feature is actually routed into output
- **State types:** linear feature directions
- **Interventions:** compare label accuracy with causal-use correlation along the proposed direction
- **Expected failure mode:** probes, PCA, or random-search directions are accepted because they decode the label even when the output ignores that feature
- **Current result:** random labelable controls have high label accuracy but low causal-use score; trained-used controls have high causal-use score

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

## Interaction recovery suite

- **File:** `mnf/benchmarks/interactions/recovery_suite.py`
- **Mechanism:** six-mechanism ecology with redundant, gated, competitive, and additive motifs
- **State types:** binary mechanism-strength states plus fuzzy atom memberships
- **Interventions:** pairwise factorial design with non-pair mechanisms held on
- **Expected failure mode:** isolated mechanism scoring misses the interaction matrix and uncertainty over interaction labels
- **Current result:** recovers all `15` pair labels with F1 `1.0` from `22` intervention states and reports bootstrap confidence intervals for gate and redundancy effects

## Noisy interaction recovery

- **File:** `mnf/benchmarks/interactions/noisy_recovery.py`
- **Mechanism:** the six-mechanism interaction recovery suite with perturbed intervention observations
- **State types:** binary mechanism-strength states
- **Interventions:** pairwise factorial design under additive observation noise
- **Expected failure mode:** exact interaction labels become unstable when contrast margins are smaller than estimation error
- **Current result:** records all-correct rate, mean F1, abstention rate, and deterministic contrast-error bounds across noise levels

## Context-stability recovery

- **File:** `mnf/benchmarks/interactions/context_stability.py`
- **Mechanism:** a `switch` changes whether the same `left/right` pair is additive or synergistic
- **State types:** binary mechanism-strength states
- **Interventions:** pairwise factorial designs under all-on and all-off background contexts
- **Expected failure mode:** a single all-on-context interaction matrix is mistaken for a context-free mechanism fact
- **Current result:** reports changed pair labels across contexts; the focal `left/right` pair changes from `additive` to `synergistic_or_gated`

## Active interaction design

- **File:** `mnf/benchmarks/interactions/active_design.py`
- **Mechanism:** active repeated-measurement recovery on the six-mechanism interaction suite
- **State types:** binary mechanism-strength states
- **Interventions:** greedy selection among pairwise factorial states, with repeats for unstable contrasts
- **Expected failure mode:** fixed one-shot designs either under-sample noisy contrasts or over-spend on already stable pairs
- **Current result:** noiseless recovery terminates at the `22`-state pairwise design; noisy recovery allocates additional measurements to unstable labels and is compared against uniform and random repeated designs at matched budgets

## Table aliasing

- **File:** `mnf/benchmarks/interactions/table_aliasing.py`
- **Mechanism:** directed gate and symmetric AND synergy induce the same four-cell output table
- **State types:** binary mechanism-strength states plus internal trace variables
- **Interventions:** pairwise factorial table plus internal gate-orientation evidence
- **Expected failure mode:** behavior-only interaction labels are mistaken for structural labels
- **Current result:** behavior-only evidence reports `ambiguous_without_internal_evidence`; internal evidence orients the directed gate as `m_to_n`

## Atom-splitting identifiability

- **File:** `mnf/benchmarks/identifiability.py`
- **Mechanism:** a split redundant route and a merged abstract route induce the same output-only response kernel
- **State types:** binary route states plus optional internal route markers
- **Interventions:** ablate route labels singly and jointly; compare output-only versus marker-augmented observables
- **Known non-identifiability:** output-only evidence cannot distinguish `split_routes` from `merged_route`
- **Expected failure mode:** a mechanism finder reports a point estimate instead of an identification set
- **Current result:** output-only max kernel distance is `0.0` with identification diameter `1.0`; adding route markers gives positive max distance and distinguishes the representatives

## No global chart

- **File:** `mnf/benchmarks/atlas/no_global_chart.py`
- **Mechanism:** the same activation coordinate needs identity or flipped gauge depending on context
- **State types:** scalar chart coordinates with a context label
- **Interventions:** chart gluing under candidate gauge transforms
- **Expected failure mode:** a single global dictionary is forced where a context-indexed atlas is required
- **Current result:** best global glue error is nonzero, while context-indexed gluing has zero error

## Higher-order interaction

- **File:** `mnf/benchmarks/interactions/higher_order.py`
- **Mechanism:** two gates jointly enable one worker mechanism
- **State types:** binary mechanism-strength states
- **Interventions:** sparse higher-order factorial design
- **Expected failure mode:** pairwise-only recovery misses that some mechanisms require higher-order contrasts
- **Current result:** pairwise-only search reports the triple as `unobserved`; order-3 search recovers a positive third-order contrast of `1.0`, and the k-way sweep recovers orders `3`, `4`, and `5`

## Tiny learned modular transformer

- **File:** `mnf/models/tiny_transformer.py`
- **Mechanism:** a one-layer CPU transformer learns modular addition over `C_7`
- **State types:** token states with cyclic counterfactual shifts
- **Interventions:** shift either input token modulo `7`; patch source-token embedding activations and final-token block activations into the base run and check output rotation consistency
- **Expected failure mode:** synthetic executable benchmarks do not exercise learned weights
- **Current result:** the 100-seed CPU sweep reaches mean final accuracy `0.9986` with mean cyclic-shift, embedding-patch, `norm1`, and `norm2` patch consistency `0.9971`; wrong-token patch controls are `0.0`

## Tiny learned redundant modular transformer

- **File:** `mnf/models/tiny_redundant.py`
- **Mechanism:** two learned transformer routes redundantly solve modular addition over `C_7`
- **State types:** route-specific token streams and logits
- **Interventions:** evaluate the full model, route-a only, route-b only, and both routes ablated
- **Expected failure mode:** single-ablation necessity rejects or underweights redundant learned routes
- **Current result:** the 100-seed CPU sweep has mean base accuracy `1.0000`, mean route-only accuracies `0.9986` and `0.9949`, mean both-routes-ablated accuracy `0.1429`, and redundancy/single-ablation-underweight rates `0.9900`

## Tiny shared-residual redundant transformer

- **File:** `mnf/models/tiny_shared_residual.py`
- **Mechanism:** one learned residual stream feeds two independently sufficient modular-addition readout routes
- **State types:** shared final residual state plus route-specific logit heads
- **Interventions:** evaluate full model, route-a only, route-b only, and both readout routes ablated
- **Expected failure mode:** explicit route separation overstates the case for learned redundancy and shared-MDL reuse
- **Current result:** the 100-seed CPU sweep has mean base accuracy `0.9994`, mean route-only accuracies `0.9994` and `0.9994`, mean both-routes-ablated accuracy `0.1429`, redundancy/single-ablation-underweight rates `1.0000`, and shared parameter gain ratio `0.4873`

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

## Known Non-Identifiability Index

| Benchmark family | Known non-identifiability or gauge |
| --- | --- |
| Cyclic weekday / modular addition | cyclic origin is arbitrary; orientation is identifiable only if the intervention generator is oriented |
| Relational lookup | categorical subject/relation/object labels are identifiable up to label permutation unless anchored by names |
| Sparse superposition | feature dictionaries are gauge-dependent under rotations that preserve sparse recovery quality |
| Hierarchical absorption | parent and child mass can be redistributed in flat sparse codes without hierarchy constraints |
| Redundant paths | single ablations cannot distinguish two redundant routes from one sufficient route |
| Gating interaction | marginal tests can confound gate strength with worker prevalence |
| Interaction phase diagram | redundancy/synergy labels vary continuously with mixture weights and thresholds |
| Context-stability recovery | pair labels depend on the background context |
| Table aliasing | output table cannot orient a directed gate without internal evidence |
| Atom-splitting identifiability | output-only response kernel cannot distinguish split versus merged redundant routes |
| No global chart | chart identity is gauge/context dependent |
| Tiny learned modular transformer | input-token mechanisms are tested at embedding and final-token block sites, but not yet decomposed into attention q/k/v or MLP submodules |
| Tiny learned redundant modular transformer | redundancy is architecture-explicit because the two learned routes are separate modules |
| Tiny shared-residual redundant transformer | redundant readout routes are still named, though they reuse one shared residual stream |
