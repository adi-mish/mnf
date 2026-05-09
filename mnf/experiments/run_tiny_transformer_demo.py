from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from mnf.models import (
    evaluate_activation_site_patching,
    evaluate_embedding_patch_interventions,
    evaluate_modular_interventions,
    torch_available,
    train_tiny_modular_addition,
)


def run(
    seeds: Sequence[int] = (0, 1, 2),
    modulus: int = 7,
    steps: int = 160,
) -> dict[str, object]:
    if not torch_available():
        return {"available": False, "reason": "optional torch dependency is not installed"}

    rows = []
    for seed in seeds:
        model, training = train_tiny_modular_addition(modulus=modulus, steps=steps, seed=seed)
        interventions = evaluate_modular_interventions(model, modulus=modulus)
        activation_patching = evaluate_embedding_patch_interventions(model, modulus=modulus)
        activation_site_patching = evaluate_activation_site_patching(model, modulus=modulus)
        rows.append(
            {
                "seed": seed,
                "training": training.as_dict(),
                "interventions": interventions,
                "activation_patching": activation_patching,
                "activation_site_patching": activation_site_patching,
            }
        )

    site_names = tuple(rows[0]["activation_site_patching"]) if rows else ()
    mean_site_patch_consistency = {
        site: float(
            np.mean(
                [
                    (
                        row["activation_site_patching"][site]["a_patch_consistency"]
                        + row["activation_site_patching"][site]["b_patch_consistency"]
                    )
                    / 2.0
                    for row in rows
                ]
            )
        )
        for site in site_names
    }
    mean_site_wrong_token_consistency = {
        site: float(
            np.mean(
                [
                    (
                        row["activation_site_patching"][site]["a_wrong_token_consistency"]
                        + row["activation_site_patching"][site]["b_wrong_token_consistency"]
                    )
                    / 2.0
                    for row in rows
                ]
            )
        )
        for site in site_names
    }

    return {
        "available": True,
        "modulus": modulus,
        "steps": steps,
        "n_seeds": len(rows),
        "rows": rows,
        "mean_final_accuracy": float(np.mean([row["training"]["final_accuracy"] for row in rows])),
        "mean_a_cyclic_shift_consistency": float(
            np.mean([row["interventions"]["a_cyclic_shift_consistency"] for row in rows])
        ),
        "mean_b_cyclic_shift_consistency": float(
            np.mean([row["interventions"]["b_cyclic_shift_consistency"] for row in rows])
        ),
        "mean_a_embedding_patch_consistency": float(
            np.mean([row["activation_patching"]["a_embedding_patch_consistency"] for row in rows])
        ),
        "mean_b_embedding_patch_consistency": float(
            np.mean([row["activation_patching"]["b_embedding_patch_consistency"] for row in rows])
        ),
        "mean_activation_site_patch_consistency": mean_site_patch_consistency,
        "mean_activation_site_wrong_token_consistency": mean_site_wrong_token_consistency,
    }
