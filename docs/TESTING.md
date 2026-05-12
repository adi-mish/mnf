# Test suites

The repository separates exact unit tests from longer CPU smoke tests.

```bash
python3 -m pytest -m "not slow"
python3 -m pytest -m "slow"
python3 -m pytest
python3 scripts/run_research_sweeps.py --config configs/cpu_quick.yaml
python3 scripts/run_research_sweeps.py --config configs/cpu_full.yaml
python3 scripts/run_research_sweeps.py --config configs/cpu_quick.yaml --skip-tiny
python3 scripts/run_tiny_transformer_sweep.py --n-seeds 20 --steps 160
python3 scripts/run_tiny_activation_recovery.py --n-seeds 3 --steps 120
python3 scripts/run_real_model_smoke.py
```

`scripts/run_research_sweeps.py` validates the generated result object against a
minimal schema before writing JSON. The quick and full configs both include the
optional tiny-transformer smoke when PyTorch is installed. Use `--skip-tiny` or
`--include-tiny` to override config-level tiny-model execution, and
`--skip-real-smoke` or `--include-real-smoke` for optional Tracr,
HF/TransformerLens CPU smokes.

Markers:

- `unit`: fast deterministic unit tests.
- `smoke`: multi-component integration checks.
- `slow`: longer CPU sweeps or expensive integration tests.
- `stochastic`: seeded stochastic checks.

The default `python3 -m pytest` still runs everything. CI can use
`python3 -m pytest -m "not slow"` if runtime needs to be capped.

## Current Durations

On 2026-05-12, `python3 -m pytest -q --durations=20` passed on the local CPU.
The slowest calls were:

| Test | Time |
| --- | ---: |
| `tests/test_interaction_suite.py::test_interaction_suite_smoke` | 19.49s |
| `tests/test_integrations.py::test_tracr_compiled_program_smoke` | 6.38s |
| `tests/test_dictionary_baselines.py::test_dictionary_baselines_and_acdc_control_expose_failures` | 1.89s |
| `tests/test_integrations.py::test_hf_transformers_smoke_boundary_is_jsonable` | 1.75s |
| `tests/test_tiny_transformer.py::test_tiny_redundant_transformer_recovers_redundant_routes_on_cpu` | 1.44s |
| `tests/test_tiny_activation_recovery.py::test_tiny_activation_recovery_smoke` | 1.20s |
