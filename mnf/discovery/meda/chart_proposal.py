from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mnf.charts.cyclic import CyclicChart
from mnf.charts.linear import LinearChart
from mnf.core.types import CyclicSpace


@dataclass(frozen=True)
class ChartProposal:
    name: str
    chart: object
    error: float
    metadata: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "error": self.error, "metadata": dict(self.metadata)}


def propose_linear_chart(activations: np.ndarray, latent_dim: int = 3, name: str = "linear") -> ChartProposal:
    chart = LinearChart.fit(activations, latent_dim=latent_dim, name=name)
    encoded = chart.encode(activations)
    decoded = chart.decode(encoded)
    error = float(np.mean((np.asarray(activations) - decoded) ** 2))
    return ChartProposal(name=name, chart=chart, error=error, metadata={"latent_dim": latent_dim})


def propose_cyclic_chart(
    activations: np.ndarray,
    labels: Sequence[Any],
    space: CyclicSpace,
    name: str = "cyclic",
) -> ChartProposal:
    chart = CyclicChart.fit(activations, labels, space)
    error = chart.intervention_error_after_rotation(activations, labels, steps=1)
    return ChartProposal(name=name, chart=chart, error=float(error), metadata={"period": space.period})
