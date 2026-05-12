from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mnf.experiments import (
    run_adversarial_controls,
    run_absorption_phase,
    run_atlas_suite,
    run_baseline_comparison,
    run_certificate_demo,
    run_cyclic_baseline_comparison,
    run_cyclic_noise_sweep,
    run_feature_baseline_suite,
    run_ground_truth_suite,
    run_identifiability_demo,
    run_induction_demo,
    run_interaction_suite,
    run_superposition_phase,
    run_tiny_shared_residual_control_demo,
    run_tiny_shared_residual_transformer_demo,
    run_tiny_redundant_transformer_demo,
    run_tiny_transformer_demo,
    run_training_emergence,
    run_transition_atom_sweep,
)
from mnf.experiments.schema import validate_research_sweeps


DEFAULT_CONFIG = {
    "absorption_n": 5000,
    "absorption_seeds": (0, 1, 2, 3, 4),
    "induction_seeds": tuple(range(20)),
    "interaction_seeds": tuple(range(20)),
    "coactivations": (0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.4, 0.7),
    "decoder_cosines": (0.0, 0.15, 0.3, 0.5, 0.7, 0.9, 1.0),
    "overlaps": (0.0, 0.25, 0.5, 0.75, 1.0),
    "redundancy_weights": (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5),
    "synergy_weights": (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5),
    "noisy_recovery_noise_levels": (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3),
    "noisy_recovery_seeds": tuple(range(200)),
    "active_design_noise_levels": (0.0, 0.005, 0.01, 0.02, 0.03, 0.04),
    "active_design_seeds": tuple(range(100)),
    "active_baseline_budgets": (22, 64, 256),
    "cyclic_noise_n": 3000,
    "cyclic_noise_seeds": (0, 1, 2, 3, 4),
    "transition_n": 6000,
    "transition_seeds": (0, 1, 2, 3, 4),
    "training_seeds": (0, 1, 2, 3, 4),
    "feature_baseline_seeds": (0, 1, 2, 3, 4),
    "include_tiny_transformer": True,
    "tiny_transformer_seeds": (0, 1, 2),
    "tiny_transformer_steps": 160,
    "tiny_redundant_transformer_seeds": (0, 1, 2),
    "tiny_redundant_transformer_steps": 160,
    "tiny_shared_residual_transformer_seeds": (0, 1, 2),
    "tiny_shared_residual_transformer_steps": 160,
    "tiny_shared_residual_control_seeds": (0, 1, 2),
    "tiny_shared_residual_control_steps": 160,
}


def _parse_config_value(value: str) -> object:
    value = value.strip()
    if value.startswith("[") or value.startswith("(") or value.startswith("{"):
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return tuple(parsed)
        return parsed
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value.strip("\"'")


def load_config(path: Path | None) -> dict[str, object]:
    config = dict(DEFAULT_CONFIG)
    if path is None:
        return config
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"invalid config line: {raw_line!r}")
        key, value = line.split(":", 1)
        key = key.strip()
        if key not in config:
            raise ValueError(f"unknown config key: {key}")
        config[key] = _parse_config_value(value)
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None, help="Optional simple YAML-style CPU sweep config")
    tiny_group = parser.add_mutually_exclusive_group()
    tiny_group.add_argument("--include-tiny", action="store_true", help="Force tiny transformer CPU smoke tests on")
    tiny_group.add_argument("--skip-tiny", action="store_true", help="Skip tiny transformer CPU smoke tests")
    parser.add_argument("--tiny-steps", type=int, default=None, help="Override all tiny transformer training steps")
    parser.add_argument("--tiny-seeds", type=int, default=None, help="Override all tiny transformer seed counts")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.include_tiny:
        config["include_tiny_transformer"] = True
    if args.skip_tiny:
        config["include_tiny_transformer"] = False
    if args.tiny_steps is not None:
        for key in (
            "tiny_transformer_steps",
            "tiny_redundant_transformer_steps",
            "tiny_shared_residual_transformer_steps",
            "tiny_shared_residual_control_steps",
        ):
            config[key] = args.tiny_steps
    if args.tiny_seeds is not None:
        seeds = tuple(range(args.tiny_seeds))
        for key in (
            "tiny_transformer_seeds",
            "tiny_redundant_transformer_seeds",
            "tiny_shared_residual_transformer_seeds",
            "tiny_shared_residual_control_seeds",
        ):
            config[key] = seeds
    out = {
        "absorption_phase": run_absorption_phase.run(
            n=int(config["absorption_n"]),
            seeds=tuple(config["absorption_seeds"]),
        ),
        "adversarial_controls": run_adversarial_controls.run(),
        "baseline_comparison": run_baseline_comparison.run(),
        "feature_baseline_suite": run_feature_baseline_suite.run(seeds=tuple(config["feature_baseline_seeds"])),
        "cyclic_baseline_comparison": run_cyclic_baseline_comparison.run(),
        "ground_truth_suite": run_ground_truth_suite.run(),
        "induction_demo": run_induction_demo.run(seeds=tuple(config["induction_seeds"])),
        "identifiability_demo": run_identifiability_demo.run(),
        "interaction_suite": run_interaction_suite.run(
            seeds=tuple(config["interaction_seeds"]),
            coactivations=tuple(config["coactivations"]),
            decoder_cosines=tuple(config["decoder_cosines"]),
            overlaps=tuple(config["overlaps"]),
            redundancy_weights=tuple(config["redundancy_weights"]),
            synergy_weights=tuple(config["synergy_weights"]),
            noisy_recovery_noise_levels=tuple(config["noisy_recovery_noise_levels"]),
            noisy_recovery_seeds=tuple(config["noisy_recovery_seeds"]),
            active_design_noise_levels=tuple(config["active_design_noise_levels"]),
            active_design_seeds=tuple(config["active_design_seeds"]),
            active_baseline_budgets=tuple(config["active_baseline_budgets"]),
        ),
        "atlas_suite": run_atlas_suite.run(),
        "certificate_demo": run_certificate_demo.run(),
        "cyclic_noise_sweep": run_cyclic_noise_sweep.run(
            n=int(config["cyclic_noise_n"]),
            seeds=tuple(config["cyclic_noise_seeds"]),
        ),
        "transition_atom_sweep": run_transition_atom_sweep.run(
            n=int(config["transition_n"]),
            seeds=tuple(config["transition_seeds"]),
        ),
        "tiny_transformer_demo": (
            run_tiny_transformer_demo.run(
                seeds=tuple(config["tiny_transformer_seeds"]),
                steps=int(config["tiny_transformer_steps"]),
            )
            if bool(config["include_tiny_transformer"])
            else {"available": False, "reason": "disabled by config"}
        ),
        "tiny_redundant_transformer_demo": (
            run_tiny_redundant_transformer_demo.run(
                seeds=tuple(config["tiny_redundant_transformer_seeds"]),
                steps=int(config["tiny_redundant_transformer_steps"]),
            )
            if bool(config["include_tiny_transformer"])
            else {"available": False, "reason": "disabled by config"}
        ),
        "tiny_shared_residual_transformer_demo": (
            run_tiny_shared_residual_transformer_demo.run(
                seeds=tuple(config["tiny_shared_residual_transformer_seeds"]),
                steps=int(config["tiny_shared_residual_transformer_steps"]),
            )
            if bool(config["include_tiny_transformer"])
            else {"available": False, "reason": "disabled by config"}
        ),
        "tiny_shared_residual_control_demo": (
            run_tiny_shared_residual_control_demo.run(
                seeds=tuple(config["tiny_shared_residual_control_seeds"]),
                steps=int(config["tiny_shared_residual_control_steps"]),
            )
            if bool(config["include_tiny_transformer"])
            else {"available": False, "reason": "disabled by config"}
        ),
        "training_emergence": run_training_emergence.run(seeds=tuple(config["training_seeds"])),
        "superposition_phase": run_superposition_phase.run(),
    }
    serializable = json.loads(json.dumps(out))
    validate_research_sweeps(serializable).raise_for_errors()
    print(json.dumps(serializable, indent=2))


if __name__ == "__main__":
    main()
