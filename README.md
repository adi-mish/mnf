# Mechanistic Normal Forms

A tested research scaffold for **Mechanistic Normal Forms (MNF)**: a theory and prototype codebase for treating mechanistic interpretability as constrained causal program induction.

The central claim is that a mechanistic explanation is not a neuron list, SAE latent list, or probe direction.  It is an executable, low-description-length, intervention-faithful, typed causal program whose interventions commute with the original model over a specified domain.

## What is included

- `mnf/core`: typed state spaces, interventions, causal programs, MNF scoring, MDL proxies, graph metrics.
- `mnf/charts`: simple linear charts, cyclic typed charts, hierarchical feature absorption utilities, MOLT-like transition atoms, and a tiny optional PyTorch TopK SAE.
- `mnf/discovery`: prototype Atlas-Causal Discovery pipeline for proposing atoms, discovering candidate edges, and fitting small causal programs.
- `mnf/benchmarks`: synthetic ground-truth systems for causal chains, gated/XOR programs, relational lookup, modular arithmetic, transition-only maps, sparse superposition, hierarchical absorption, cyclic weekday features, shortcut controls, memorization controls, and random-vs-trained controls.
- `mnf/experiments`: runnable demos and CPU-only phase sweeps for the benchmark families.
- `tests`: 28 passing tests that exercise all core components.
- `docs/THEORY.md`: detailed theory.
- `docs/RESEARCH_PROGRAM.md`: aggressive NeurIPS-scale research plan.
- `docs/demo_results.json`: output of the demos on the current environment.
- `docs/research_sweeps.json`: CPU-generated phase/sweep outputs for local benchmark experiments.

## Quick start

```bash
cd mechanistic-normal-forms
python -m pip install -e '.[dev]'
pytest -q
python scripts/run_all_demos.py
python scripts/run_research_sweeps.py
```

The repo was tested in the current environment with Python 3.13.2.  All 28 tests passed.

## What this is not

This is not a complete LLM interpretability pipeline.  It does not download models, train production-scale SAEs, run TransformerLens hooks, or reproduce Anthropic-style attribution graphs.  It is a rigorous scaffold: definitions, scoring, toy ground-truth benchmarks, typed charts, transition atoms, synthetic falsification tests, and a plan for scaling to real models.

## Design principle

Every explanatory object must answer four questions:

1. What state space does it live in?
2. How is it read from activations?
3. How is it intervened on in activation space?
4. What downstream causal role does it play?

If it cannot answer those questions, it may be decodable, but it is not yet mechanistic.
