from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path


def certificate_pareto_svg(report: Mapping[str, object], path: str | Path) -> None:
    """Write a small dependency-free SVG Pareto plot for certificate reports."""

    certs = report.get("certificates", {})
    if not isinstance(certs, Mapping):
        raise ValueError("report must contain a certificates object")
    names = list(certs)
    if not names:
        raise ValueError("at least one certificate is required")
    xs = [float(certs[name]["shared_description_length"]) for name in names]  # type: ignore[index]
    ys = [float(certs[name]["intervention_error"]) for name in names]  # type: ignore[index]
    frontier = set(int(i) for i in report.get("pareto_indices", ()))
    width, height = 640, 420
    margin = 56
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax == xmin:
        xmax = xmin + 1.0
    if ymax == ymin:
        ymax = ymin + 1.0

    def px(x: float) -> float:
        return margin + (x - xmin) / (xmax - xmin) * (width - 2 * margin)

    def py(y: float) -> float:
        return height - margin - (y - ymin) / (ymax - ymin) * (height - 2 * margin)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#222"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#222"/>',
        f'<text x="{width/2}" y="{height-14}" text-anchor="middle" font-family="sans-serif" font-size="14">Shared description length</text>',
        f'<text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle" font-family="sans-serif" font-size="14">Intervention error</text>',
    ]
    for i, name in enumerate(names):
        color = "#1f77b4" if i in frontier else "#999"
        radius = 6 if i in frontier else 4
        x, y = px(xs[i]), py(ys[i])
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{color}"/>')
        parts.append(
            f'<text x="{x + 8:.2f}" y="{y - 8:.2f}" font-family="sans-serif" font-size="12">{name}</text>'
        )
    parts.append("</svg>")
    Path(path).write_text("\n".join(parts) + "\n")
