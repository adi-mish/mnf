from __future__ import annotations

from collections.abc import Mapping, Sequence

from mnf.baselines import acdc_redundancy_failure_demo
from mnf.discovery.meda import discover_meda
from mnf.models import (
    collect_tiny_hook_activations,
    evaluate_redundant_route_interventions,
    modular_addition_dataset,
    torch_available,
    train_tiny_modular_addition,
    train_tiny_redundant_modular_transformer,
)


def _route_accuracy_behavior(model, modulus: int):
    x, y = modular_addition_dataset(modulus=modulus)

    def behavior(state: Mapping[str, bool]) -> float:
        mask = (1.0 if state["route_a"] else 0.0, 1.0 if state["route_b"] else 0.0)
        with __import__("torch").no_grad():
            pred = model(x, route_mask=mask).argmax(dim=-1)
            return float((pred == y).float().mean().detach().cpu())

    return behavior


def run(
    seeds: Sequence[int] = (0,),
    modulus: int = 7,
    steps: int = 120,
) -> dict[str, object]:
    if not torch_available():
        return {"available": False, "reason": "optional torch dependency is not installed"}

    rows = []
    for seed in seeds:
        modular_model, modular_training = train_tiny_modular_addition(modulus=modulus, steps=steps, seed=seed)
        tokens, _ = modular_addition_dataset(modulus=modulus)
        hook_activations = collect_tiny_hook_activations(modular_model, tokens[:4])
        hook_shapes = {name: list(value.shape) for name, value in hook_activations.items() if hasattr(value, "shape")}

        redundant_model, redundant_training = train_tiny_redundant_modular_transformer(
            modulus=modulus,
            steps=steps,
            seed=seed,
        )
        interventions = evaluate_redundant_route_interventions(redundant_model, modulus=modulus)
        meda = discover_meda(
            names=("route_a", "route_b"),
            behavior_fn=_route_accuracy_behavior(redundant_model, modulus),
            memberships=({"shared_task": 1.0, "route_a": 1.0}, {"shared_task": 1.0, "route_b": 1.0}),
        )
        rows.append(
            {
                "seed": seed,
                "modular_training": modular_training.as_dict(),
                "hook_shapes": hook_shapes,
                "redundant_training": redundant_training.as_dict(),
                "route_interventions": interventions,
                "meda": meda.as_dict(),
            }
        )

    redundancy_labels = [
        row["meda"]["structural"]["route_a::route_b"]["structural_label"]  # type: ignore[index]
        for row in rows
    ]
    return {
        "available": True,
        "modulus": modulus,
        "steps": steps,
        "n_seeds": len(rows),
        "rows": rows,
        "hook_sites": sorted(rows[0]["hook_shapes"]) if rows else [],
        "meda_redundancy_recovered_rate": float(sum(label == "redundant" for label in redundancy_labels) / len(rows)),
        "single_ablation_underweights_rate": float(
            sum(bool(row["route_interventions"]["single_ablation_underweights"]) for row in rows) / len(rows)
        ),
        "acdc_redundancy_failure": acdc_redundancy_failure_demo(),
    }
