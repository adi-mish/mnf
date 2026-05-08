# Aggressive research program

## Target paper

**Mechanistic Normal Forms: Natural Causal Abstractions for Interpreting Neural Networks**

Core claim: mechanistic interpretability should recover minimum-description-length natural causal programs, not merely label neurons or sparse latents.

## Landmark contribution stack

A NeurIPS-best-paper-caliber version needs four contributions in one paper:

1. **Theory:** formal definitions of natural causal abstraction, typed variables, transition atoms, MNF score, and falsifiability criteria.
2. **Theorems:** sparse packing, feature absorption, typed-feature identifiability up to gauge, and vacuity of unconstrained abstraction.
3. **Benchmark:** MechanismLab, a ground-truth suite spanning scalar, hierarchical, cyclic, relational, transition-only, compiled, trained, and random-control systems.
4. **Algorithm:** Atlas-Causal Discovery, which recovers executable mechanisms and beats probes, flat SAEs, and component-level circuit discovery on intervention prediction.

## Phase 0: repository hardening

- Add CI.
- Add deterministic seeds for all tests.
- Add type checking with pyright or mypy.
- Add richer plotting scripts.
- Add experiment config files.
- Add benchmark cards documenting each synthetic mechanism.

## Phase 1: formal theory

Write a formal paper draft with:

- domain-relative explanation definition;
- intervention algebra;
- naturalness constraints;
- typed state spaces and gauge symmetries;
- distinction between decodable / represented / used / mechanistic;
- state atoms and transition atoms;
- MDL objective;
- theorem statements.

First proofs to complete:

1. Sparse packing theorem in a linear sparse-feature world.
2. Absorption theorem for parent-child features under L0/L1 sparsity.
3. Identifiability up to gauge for scalar, cyclic, and categorical typed variables with interventions.
4. Non-identifiability/vacuity under arbitrary nonlinear alignments.

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
- trained small transformers;
- Tracr/RASP compiled transformers if dependencies permit.

Metrics:

- observation error;
- intervention prediction error;
- node F1 against ground truth;
- edge F1 against ground truth;
- description length;
- invariance under paraphrase/environment/checkpoint;
- false mechanism rate on random controls.

## Phase 3: Atlas-Causal Discovery v1

Upgrade the prototype pipeline:

- proposal stage with scalar SAEs, TopK SAEs, cyclic charts, hierarchy charts, supervised contrastive directions, and transition atoms;
- validation stage with activation patching, path patching, and counterfactual interventions;
- pruning stage with MDL and invariance penalties;
- output stage producing an executable mechanism DSL.

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
8. Training-dynamics result: MNF progress predicts behavioral emergence.

## Phase 7: what to cut if time is short

The highest-value core is:

1. MNF definitions and scoring.
2. MechanismLab ground-truth benchmark.
3. Absorption theorem + experiment.
4. Cyclic typed-feature experiment.
5. Random-control experiment.
6. One real-model case study.

This still makes a strong paper because it attacks the foundation of what counts as a mechanistic explanation.
