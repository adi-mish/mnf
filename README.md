# Mechanistic Normal Forms

A tested research scaffold for **Mechanistic Normal Forms (MNF)**: a theory and prototype codebase for treating mechanistic interpretability as constrained causal program induction and response-kernel factorization.

The central claim is that a mechanistic explanation is not a neuron list, SAE latent list, or probe direction.  It is an executable, low-description-length, intervention-faithful, typed causal program whose interventions commute with the original model over a specified domain and whose remaining non-identifiability is explicitly reported.

## What is included

- `mnf/core`: typed state spaces, interventions, causal programs, MNF scoring, MDL proxies, graph metrics.
- `mnf/mechanisms`: mechanism proposals, soft atom-membership coordinates, mechanism-strength interventions, shared atom accounting, and ecosystem-level shared MDL.
- `mnf/interactions`: pairwise factorial effects, redundancy/gating/synergy metrics, capacity competition, support, gradient coupling, Shapley allocation, and coupled dynamics.
- `mnf/semantics`: finite interventional response-kernel tables, kernel distances, naturalness profiles, equivalence checks, and identification-set construction.
- `mnf/certificates`: certificate vectors, identification sets, Pareto ordering, thresholds, and report helpers for mechanism claims.
- `mnf/atlas`: context-indexed chart and gauge/gluing utilities for interventional mechanism atlases.
- `mnf/charts`: simple linear charts, cyclic typed charts, hierarchical feature absorption utilities, MOLT-like transition atoms, and a tiny optional PyTorch TopK SAE.
- `mnf/models`: optional tiny CPU transformer components for learned-mechanism smoke tests.
- `mnf/baselines`: lightweight probe, PCA, random-feature, shortcut-selection, and scalar/PCA-style chart comparisons.
- `mnf/discovery`: prototype Atlas-Causal Discovery pipeline plus a minimal Mechanism Ecology Discovery adapter for pairwise interaction recovery.
- `mnf/benchmarks`: synthetic ground-truth systems for causal chains, gated/XOR programs, induction match-copy, relational lookup, modular arithmetic, transition-only maps, sparse superposition, hierarchical absorption, cyclic weekday features, shortcut controls, memorization controls, training-emergence controls, interaction/ecology controls, and random-vs-trained controls.
- `mnf/experiments`: runnable demos and CPU-only phase sweeps for the benchmark families.
- `tests`: passing tests that exercise all core components, including slow CPU smoke tests.
- `docs/THEORY.md`: detailed theory.
- `docs/INTERVENTIONAL_ATLASES.md`: iMNF/IMA response-kernel atlas, certificate-vector, identification-set, gauge/gluing, and structural-interaction formulation.
- `docs/FORMAL_THEORY.md`: theorem candidates and proof sketches connected to benchmarks.
- `docs/INTERACTION_CALCULUS.md`: executable iMNF interaction metrics, intervention designs, uncertainty, and recovery metrics.
- `docs/THEOREMS.md`: proof-obligation package for the iMNF claims and their benchmark witnesses.
- `docs/THEORY_AUDIT.md`: consistency and rigor audit for the theory documents.
- `docs/ADVERSARIAL_REVIEW.md`: adversarial review of the current theory claims, attack surfaces, and required evidence.
- `docs/TESTING.md`: quick, slow, smoke, and stochastic test-suite conventions.
- `docs/RESEARCH_PROGRAM.md`: aggressive NeurIPS-scale research plan.
- `docs/BENCHMARK_CARDS.md`: documented MechanismLab benchmark cards.
- `docs/demo_results.json`: output of the demos on the current environment.
- `docs/research_sweeps.json`: CPU-generated phase/sweep outputs for local benchmark experiments.
- `docs/RESULTS_SUMMARY.md`: compact generated summary of the sweep outputs.
- `docs/AGGRESSIVE_CPU_SUMMARY.md`: 1,000-seed CPU robustness summary for noisy/active interaction recovery.
- `docs/TINY_TRANSFORMER_CPU_SUMMARY.md`: 20-seed CPU robustness summary for the learned tiny modular-addition transformer.

## Quick start

```bash
cd mechanistic-normal-forms
python -m pip install -e '.[dev]'
pytest -q
pytest -q -m "not slow"
python scripts/run_all_demos.py
python scripts/run_research_sweeps.py --config configs/cpu_full.yaml
python scripts/write_results_summary.py
```

The repo was tested in the current environment with Python 3.13.2.

## What this is not

This is not a complete LLM interpretability pipeline.  It does not download large models, train production-scale SAEs, run TransformerLens hooks, or reproduce Anthropic-style attribution graphs.  It is a rigorous scaffold: definitions, scoring, toy ground-truth benchmarks, typed charts, transition atoms, synthetic falsification tests, a tiny learned-transformer smoke test, and a plan for scaling to real models.

## Design principle

Every explanatory object must answer five questions:

1. What state space does it live in?
2. How is it read from activations?
3. How is it intervened on in activation space?
4. What downstream causal role does it play?
5. Which alternatives remain indistinguishable under the current interventions?

If it cannot answer those questions, it may be decodable, but it is not yet mechanistic.

The iMNF extension adds a sixth question: how does this mechanism interact with
other mechanisms through redundancy, gating, support, shared atoms, capacity
competition, or training-time coupling?
