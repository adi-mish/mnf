from __future__ import annotations

from mnf.atlas import MechanismChart, affine_gauge, best_gluing_report, gluing_error, identity_gauge


def _samples() -> tuple[dict[str, object], ...]:
    return tuple(
        {"activation": float(value), "context": context}
        for context in ("identity", "flip")
        for value in (0.0, 0.25, 0.5, 0.75, 1.0)
    )


def _source_chart() -> MechanismChart:
    return MechanismChart(
        "source",
        ("z",),
        read=lambda sample: {"z": float(sample["activation"])},
        domain_label="default",
    )


def _target_chart() -> MechanismChart:
    def read(sample: dict[str, object]) -> dict[str, float]:
        value = float(sample["activation"])
        if sample["context"] == "flip":
            value = 1.0 - value
        return {"z": value}

    return MechanismChart("target", ("z",), read=read, domain_label="default")


def no_global_chart_metrics() -> dict[str, object]:
    samples = _samples()
    source = _source_chart()
    target = _target_chart()
    identity = identity_gauge()
    flip = affine_gauge(scale=-1.0, offset=1.0, name="flip")
    identity_report = gluing_error(source, target, identity, samples)
    flip_report = gluing_error(source, target, flip, samples)
    best_global = best_gluing_report(source, target, (identity, flip), samples)
    context_reports = [
        gluing_error(source, target, identity, [sample for sample in samples if sample["context"] == "identity"]),
        gluing_error(source, target, flip, [sample for sample in samples if sample["context"] == "flip"]),
    ]
    context_indexed_error = sum(report.error * report.n_overlap for report in context_reports) / sum(
        report.n_overlap for report in context_reports
    )
    return {
        "identity_glue_error": identity_report.error,
        "flip_glue_error": flip_report.error,
        "best_global_gauge": best_global.gauge,
        "best_global_glue_error": best_global.error,
        "context_indexed_glue_error": float(context_indexed_error),
        "requires_context_indexed_gauge": bool(context_indexed_error < best_global.error),
        "n_samples": len(samples),
    }
