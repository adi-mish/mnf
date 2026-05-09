from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from mnf.models import evaluate_modular_interventions, torch_available, train_tiny_modular_addition


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
        rows.append(
            {
                "seed": seed,
                "training": training.as_dict(),
                "interventions": interventions,
            }
        )

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
    }
