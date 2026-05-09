from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from mnf.atlas.chart import MechanismChart, state_distance
from mnf.atlas.gauge import GaugeTransform


@dataclass(frozen=True)
class GluingReport:
    source_chart: str
    target_chart: str
    gauge: str
    error: float
    n_overlap: int

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "source_chart": self.source_chart,
            "target_chart": self.target_chart,
            "gauge": self.gauge,
            "error": float(self.error),
            "n_overlap": int(self.n_overlap),
        }


def gluing_error(
    source: MechanismChart,
    target: MechanismChart,
    gauge: GaugeTransform,
    samples: Sequence[Mapping[str, object]],
    keys: Sequence[str] | None = None,
) -> GluingReport:
    errs = []
    for sample in samples:
        source_state = gauge.apply(source.encode(sample))
        target_state = target.encode(sample)
        use_keys = tuple(keys or sorted(set(source_state) & set(target_state)))
        errs.append(state_distance(source_state, target_state, use_keys))
    return GluingReport(
        source_chart=source.name,
        target_chart=target.name,
        gauge=gauge.name,
        error=float(np.mean(errs)) if errs else 0.0,
        n_overlap=len(samples),
    )


def best_gluing_report(
    source: MechanismChart,
    target: MechanismChart,
    gauges: Sequence[GaugeTransform],
    samples: Sequence[Mapping[str, object]],
    keys: Sequence[str] | None = None,
) -> GluingReport:
    if not gauges:
        raise ValueError("at least one gauge is required")
    reports = [gluing_error(source, target, gauge, samples, keys=keys) for gauge in gauges]
    return min(reports, key=lambda report: report.error)
