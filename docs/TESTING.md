# Test suites

The repository separates exact unit tests from longer CPU smoke tests.

```bash
python3 -m pytest -m "not slow"
python3 -m pytest -m "slow"
python3 -m pytest
python3 scripts/run_research_sweeps.py --config configs/cpu_quick.yaml
python3 scripts/run_research_sweeps.py --config configs/cpu_full.yaml
python3 scripts/run_tiny_transformer_sweep.py --n-seeds 20 --steps 160
```

`scripts/run_research_sweeps.py` validates the generated result object against a
minimal schema before writing JSON. The quick and full configs both include the
optional tiny-transformer smoke when PyTorch is installed.

Markers:

- `unit`: fast deterministic unit tests.
- `smoke`: multi-component integration checks.
- `slow`: longer CPU sweeps or expensive integration tests.
- `stochastic`: seeded stochastic checks.

The default `python3 -m pytest` still runs everything. CI can use
`python3 -m pytest -m "not slow"` if runtime needs to be capped.
