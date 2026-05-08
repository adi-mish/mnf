from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from mnf.interactions.capacity import capacity_competition_from_matrices


def capacity_competition_score(coactivation: float, decoder_cosine: float, overlap: float = 1.0) -> float:
    memberships_m = np.array([1.0, overlap], dtype=float)
    memberships_n = np.array([overlap, 1.0], dtype=float)
    q = np.full((2, 2), float(coactivation), dtype=float)
    gram_squared = np.array(
        [
            [float(decoder_cosine) ** 2, float(decoder_cosine) ** 2],
            [float(decoder_cosine) ** 2, float(decoder_cosine) ** 2],
        ]
    )
    return capacity_competition_from_matrices(memberships_m, memberships_n, q, gram_squared)


def capacity_competition_sweep(
    coactivations: Sequence[float] = (0.01, 0.05, 0.15, 0.4),
    decoder_cosines: Sequence[float] = (0.0, 0.25, 0.5, 0.9),
    overlaps: Sequence[float] = (0.0, 0.5, 1.0),
) -> dict[str, object]:
    rows = []
    for q in coactivations:
        for cosine in decoder_cosines:
            for overlap in overlaps:
                score = capacity_competition_score(q, cosine, overlap)
                rows.append(
                    {
                        "coactivation": float(q),
                        "decoder_cosine": float(cosine),
                        "overlap": float(overlap),
                        "capacity_competition": float(score),
                        "predicted_error_increase": float(0.2 * score),
                    }
                )
    return {"rows": rows}
