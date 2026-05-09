from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


JSONScalar = str | int | float | bool | None
JSONValue = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


@dataclass(frozen=True)
class SchemaValidation:
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.errors:
            joined = "\n".join(self.errors)
            raise ValueError(f"result schema validation failed:\n{joined}")

    def as_dict(self) -> dict[str, object]:
        return {"ok": self.ok, "errors": list(self.errors)}


def _is_jsonable(value: Any, path: str, errors: list[str]) -> None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if isinstance(value, list):
        for idx, item in enumerate(value):
            _is_jsonable(item, f"{path}[{idx}]", errors)
        return
    if isinstance(value, tuple):
        errors.append(f"{path}: tuple should be converted to list before writing JSON")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                errors.append(f"{path}: object key {key!r} is not a string")
            _is_jsonable(item, f"{path}.{key}", errors)
        return
    errors.append(f"{path}: unsupported value type {type(value).__name__}")


def validate_jsonable_result(data: Mapping[str, Any]) -> SchemaValidation:
    errors: list[str] = []
    _is_jsonable(data, "$", errors)
    return SchemaValidation(tuple(errors))


def _require_path(data: Mapping[str, Any], path: tuple[str, ...], errors: list[str]) -> Any:
    current: Any = data
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            errors.append("$." + ".".join(path) + " is required")
            return None
        current = current[part]
    return current


def _require_numeric(data: Mapping[str, Any], path: tuple[str, ...], errors: list[str]) -> None:
    value = _require_path(data, path, errors)
    if value is not None and not isinstance(value, (int, float)):
        errors.append("$." + ".".join(path) + f" must be numeric, got {type(value).__name__}")


def validate_research_sweeps(data: Mapping[str, Any]) -> SchemaValidation:
    errors: list[str] = list(validate_jsonable_result(data).errors)
    required = (
        "absorption_phase",
        "adversarial_controls",
        "atlas_suite",
        "baseline_comparison",
        "certificate_demo",
        "cyclic_baseline_comparison",
        "cyclic_noise_sweep",
        "feature_baseline_suite",
        "ground_truth_suite",
        "induction_demo",
        "interaction_suite",
        "superposition_phase",
        "training_emergence",
        "transition_atom_sweep",
    )
    for key in required:
        if key not in data:
            errors.append(f"$.{key} is required")

    _require_numeric(data, ("interaction_suite", "higher_order", "third_order_effect"), errors)
    _require_path(data, ("interaction_suite", "higher_order", "triple_search", "label_counts"), errors)
    _require_path(data, ("interaction_suite", "context_stability_sweep", "rows"), errors)
    _require_numeric(data, ("atlas_suite", "no_global_chart", "best_global_glue_error"), errors)
    _require_numeric(data, ("atlas_suite", "no_global_chart", "context_indexed_glue_error"), errors)
    _require_path(data, ("certificate_demo", "report", "pareto_indices"), errors)
    _require_numeric(data, ("feature_baseline_suite", "mean_random_false_mechanism_rate"), errors)
    _require_numeric(data, ("feature_baseline_suite", "mean_trained_causal_use_score"), errors)
    _require_path(data, ("tiny_transformer_demo", "available"), errors)
    if data.get("tiny_transformer_demo", {}).get("available"):
        _require_numeric(data, ("tiny_transformer_demo", "mean_final_accuracy"), errors)
        _require_numeric(data, ("tiny_transformer_demo", "mean_a_cyclic_shift_consistency"), errors)
        _require_numeric(data, ("tiny_transformer_demo", "mean_b_cyclic_shift_consistency"), errors)
    return SchemaValidation(tuple(errors))
