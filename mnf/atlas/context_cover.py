from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from mnf.atlas.chart import MechanismChart


@dataclass(frozen=True)
class ContextCover:
    charts: tuple[MechanismChart, ...]

    def chart_names(self) -> tuple[str, ...]:
        return tuple(chart.name for chart in self.charts)

    def select(self, sample: Mapping[str, object]) -> tuple[MechanismChart, ...]:
        label = sample.get("context")
        return tuple(
            chart
            for chart in self.charts
            if chart.domain_label == "default" or chart.domain_label == label
        )


def cover_counts(cover: ContextCover, samples: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts = {name: 0 for name in cover.chart_names()}
    for sample in samples:
        for chart in cover.select(sample):
            counts[chart.name] += 1
    return counts
