# Aggressive research program

## Target paper

**Interventional Mechanism Atlases: Causal Mechanism Ecologies in Neural Networks**

Core claim: mechanistic interpretability should recover context-indexed,
gauge-aware, graded, interacting, minimum-description-length causal mechanism
atlases that factorize a model's interventional response kernel, not merely
label neurons, sparse latents, or isolated circuits.

## Landmark contribution stack

A NeurIPS-best-paper-caliber version needs four contributions in one paper:

1. **Theory:** natural causal abstraction, response-kernel factorization, typed variables, transition atoms, gauge/gluing, identification sets, certificate vectors, shared MDL, interaction terms, and falsifiability criteria.
2. **Theorems and propositions:** feature absorption, typed-feature identifiability up to gauge, response-kernel atom-splitting ambiguity, behavior-table structural ambiguity, context-gluing obstruction, isolated-circuit fallacy, gating non-identifiability under marginal interventions, shared-MDL preference, developmental bootstrapping, sparse mechanism packing scores, and vacuity of unconstrained abstraction.
3. **Benchmark:** MechanismLab 4.0, a ground-truth suite spanning scalar, hierarchical, cyclic, relational, transition-only, atlas/gluing, structural-aliasing, response-kernel non-identifiability, interaction/ecology, compiled, trained, and random-control systems.
4. **Algorithm:** Mechanism Ecology Discovery with Atlases, which recovers executable mechanisms, context-indexed charts, certificates, and interaction matrices and beats probes, flat SAEs, and component-level circuit discovery on intervention prediction.

## Phase 0: repository hardening

- Add CI.
- Add deterministic seeds for all tests.
- Add pytest markers for quick, slow, smoke, and stochastic tests.
- Add type checking with pyright or mypy.
- Add richer plotting scripts.
- Add experiment config files.
- Add result schemas with validation. Current status: `scripts/run_research_sweeps.py`
  validates required top-level sections and key atlas/certificate/interaction
  fields before writing JSON.
- Add benchmark cards documenting each synthetic mechanism.
- Add lightweight feature baselines. Current status: linear probe, PCA-first,
  and random-search directions are compared on random-labelable versus
  trained-used controls.

## Phase 1: formal theory

Write a formal paper draft with:

- domain-relative explanation definition;
- intervention algebra;
- naturalness constraints;
- typed state spaces and gauge symmetries;
- context-indexed atlas charts and gluing errors;
- distinction between decodable / represented / used / mechanistic;
- state atoms and transition atoms;
- fuzzy mechanism membership;
- redundancy, gating, support, capacity competition, gradient coupling, and developmental coupling;
- shared-MDL accounting for atom reuse;
- MDL objective;
- theorem statements.
- certificate-vector acceptance and Pareto comparison.
- response-kernel semantics and identification-set reporting.

First proofs to complete:

1. Sparse packing theorem in a linear sparse-feature world.
2. Absorption theorem for parent-child features under L0/L1 sparsity.
3. Identifiability up to gauge for scalar, cyclic, and categorical typed variables with interventions.
4. Non-identifiability/vacuity under arbitrary nonlinear alignments.
5. Isolated-circuit fallacy under redundant paths.
6. Gating non-identifiability under marginal interventions.
7. Behavior-table ambiguity under identical four-cell outputs.
8. Context-gluing obstruction for non-global charts.
9. Shared-MDL preference for reusable atoms.
10. Response-kernel atom-splitting non-identifiability under weak observables.

## Phase 2: MechanismLab benchmark

Extend `mnf/benchmarks` into a real benchmark package.

Required tasks:

- scalar causal chains;
- XOR/gated mechanisms;
- sparse superposition;
- hierarchical absorption;
- cyclic weekdays/months/modular arithmetic;
- relational subject-relation-object lookup;
- transition-only synthetic MLPs;
- random controls;
- trained small transformers. Current status: an optional CPU tiny transformer
  learns modular addition over `C_7` and is evaluated by cyclic input-shift
  interventions plus embedding/final-token block activation patching; the
  100-seed local CPU sweep reaches mean final accuracy `0.9986` with mean
  cyclic-shift and activation-patch consistency about `0.9971`.
- trained redundant small transformers. Current status: a two-route CPU
  transformer learns modular addition with either route sufficient; the 100-seed
  local CPU sweep reaches base accuracy `1.0000`, route-only accuracies `0.9986`
  and `0.9949`, dual-ablation drop `0.8571`, and single-ablation underweight
  rate `0.9900`.
- trained shared-residual redundant small transformers. Current status: one CPU
  transformer reuses a shared residual stream for two independently sufficient
  readout routes; the 100-seed local CPU sweep reaches base accuracy `0.9994`,
  route-only accuracies `0.9994` and `0.9994`, dual-ablation drop `0.8565`, and
  shared parameter gain ratio `0.4873`.
- redundant-path, gating, support, and capacity-competition mechanism ecologies;
- coupled training dynamics, including developmental bootstrap and mechanism death;
- atom-splitting response-kernel non-identifiability with explicit
  `IdentificationSet` output;
- Tracr/RASP compiled transformers if dependencies permit.

Metrics:

- observation error;
- intervention prediction error;
- node F1 against ground truth;
- edge F1 against ground truth;
- description length;
- invariance under paraphrase/environment/checkpoint;
- false mechanism rate on random controls.
- identification-set diameter and distinguishable/indistinguishable evidence;
- interaction matrix recovery error;
- shared-MDL gain against independent mechanism descriptions.

## Phase 3: Mechanism Ecology Discovery v1

Upgrade the prototype pipeline:

- proposal stage with scalar SAEs, TopK SAEs, cyclic charts, hierarchy charts, supervised contrastive directions, and transition atoms;
- fuzzy mechanism proposal using masks over typed atoms;
- validation stage with activation patching, path patching, and counterfactual interventions;
- pairwise factorial intervention stage for redundancy, synergy, gating, and support;
- bootstrap uncertainty estimates for interaction contrasts;
- deterministic contrast-error bounds and noisy recovery phase sweeps;
- active repeated-measurement design for unstable interaction contrasts;
- capacity-competition and gradient-coupling estimation;
- recovery metrics for pair labels and interaction matrices;
- pruning stage with MDL and invariance penalties;
- shared-MDL pruning stage over atoms, mechanisms, and interactions;
- output stage producing an executable mechanism-ecosystem DSL.

## Phase 4: real-model integration

Add optional integrations:

- TransformerLens activation hooks;
- SAE Lens / SAEBench feature loading;
- Gemma Scope feature loading;
- Neuronpedia metadata loading;
- ACDC baseline wrapper;
- transcoder / CLT baseline wrapper where available.

Initial real-model case studies:

- GPT-2 small IOI;
- GPT-2 small greater-than;
- induction heads in small attention-only models;
- weekday/month cyclic features;
- refusal direction vs distributed refusal gates;
- factual recall / ROME-style localization.

## Phase 5: adversarial evaluation

Build controls that make ordinary interpretability methods look good but should fail MNF:

- random initialized transformers;
- label-shuffled feature datasets;
- noncausal probes;
- high-complexity alignment maps;
- spurious correlation removal;
- shortcut/memorization mechanisms;
- adversarial paraphrase sets.

## Phase 6: paper experiments

Minimum set for a serious submission:

1. Ground-truth recovery on MechanismLab.
2. Superposition phase diagram matching theorem.
3. Absorption phase diagram and hierarchical fix.
4. Cyclic features: scalar dictionaries fragment, typed charts recover.
5. Random-control separation: labelability vs mechanistic use.
6. Transition atoms outperform state-only explanations on MLP-heavy synthetic tasks.
7. Real-model case studies showing lower intervention error and shorter descriptions than baselines.
8. Interaction result: factorial interventions recover redundancy and gating where single ablations fail.
9. Training-dynamics result: MNF progress predicts behavioral emergence and coupled dynamics predict bootstrap or suppression.

## Phase 7: what to cut if time is short

The highest-value core is:

1. MNF definitions and scoring.
2. MechanismLab ground-truth benchmark.
3. Absorption theorem + experiment.
4. Cyclic typed-feature experiment.
5. Random-control experiment.
6. One real-model case study.
7. One interaction result showing isolated-circuit failure.

This still makes a strong paper because it attacks the foundation of what counts as a mechanistic explanation.
