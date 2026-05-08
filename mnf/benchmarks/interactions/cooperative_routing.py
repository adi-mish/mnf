from __future__ import annotations

from mnf.interactions.mediation import normalized_support_delta, support_delta


def cooperative_routing_metrics() -> dict[str, float]:
    intact_mechanisticity = 0.82
    without_route_mechanisticity = 0.31
    return {
        "intact_mechanisticity": intact_mechanisticity,
        "without_route_mechanisticity": without_route_mechanisticity,
        "support_delta": support_delta(intact_mechanisticity, without_route_mechanisticity),
        "normalized_support_delta": normalized_support_delta(intact_mechanisticity, without_route_mechanisticity),
    }
