from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from mnf.models.tiny_transformer import _require_torch, modular_addition_dataset, torch_available


_torch = None
try:
    import torch as _torch
except ImportError:  # pragma: no cover - torch availability is tested through torch_available
    pass

_BaseModule = _torch.nn.Module if _torch is not None else object


class TinySharedResidualRedundantTransformer(_BaseModule):
    """Shared-encoder transformer with two redundant readout routes."""

    def __init__(
        self,
        modulus: int = 7,
        d_model: int = 32,
        n_heads: int = 4,
        d_ff: int = 64,
    ) -> None:
        torch = _require_torch()
        super().__init__()
        self.modulus = modulus
        self.embedding = torch.nn.Embedding(modulus + 1, d_model)
        self.position = torch.nn.Parameter(torch.zeros(3, d_model))
        layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            batch_first=True,
            dropout=0.0,
            activation="gelu",
        )
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=1)
        self.route_a_output = torch.nn.Linear(d_model, modulus)
        self.route_b_output = torch.nn.Linear(d_model, modulus)

    def shared_state(self, tokens: Any) -> Any:
        hidden = self.embedding(tokens) + self.position.unsqueeze(0)
        encoded = self.encoder(hidden)
        return encoded[:, -1]

    def route_logits(self, tokens: Any) -> tuple[Any, Any]:
        state = self.shared_state(tokens)
        return self.route_a_output(state), self.route_b_output(state)

    def forward(self, tokens: Any, route_mask: tuple[float, float] = (1.0, 1.0)) -> Any:
        torch = _require_torch()
        logits_a, logits_b = self.route_logits(tokens)
        mask_a, mask_b = route_mask
        active = float(mask_a + mask_b)
        if active == 0.0:
            return torch.zeros_like(logits_a)
        return (mask_a * logits_a + mask_b * logits_b) / active

    def parameter_accounting(self) -> dict[str, int | float]:
        shared_core = _parameter_count(self.embedding) + self.position.numel() + _parameter_count(self.encoder)
        route_a = _parameter_count(self.route_a_output)
        route_b = _parameter_count(self.route_b_output)
        shared_total = shared_core + route_a + route_b
        independent_total = 2 * shared_core + route_a + route_b
        return {
            "shared_core_parameters": int(shared_core),
            "route_a_parameters": int(route_a),
            "route_b_parameters": int(route_b),
            "shared_total_parameters": int(shared_total),
            "independent_total_parameters": int(independent_total),
            "shared_parameter_gain": int(independent_total - shared_total),
            "shared_parameter_gain_ratio": float((independent_total - shared_total) / independent_total),
        }


def _parameter_count(module: Any) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


@dataclass(frozen=True)
class TinySharedResidualTrainingResult:
    final_loss: float
    base_accuracy: float
    route_a_only_accuracy: float
    route_b_only_accuracy: float

    def as_dict(self) -> dict[str, float]:
        return {
            "final_loss": float(self.final_loss),
            "base_accuracy": float(self.base_accuracy),
            "route_a_only_accuracy": float(self.route_a_only_accuracy),
            "route_b_only_accuracy": float(self.route_b_only_accuracy),
        }


def train_tiny_shared_residual_redundant_transformer(
    modulus: int = 7,
    steps: int = 160,
    lr: float = 3e-3,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[TinySharedResidualRedundantTransformer, TinySharedResidualTrainingResult]:
    torch = _require_torch()
    torch.manual_seed(seed)
    model = TinySharedResidualRedundantTransformer(modulus=modulus).to(device)
    x, y = modular_addition_dataset(modulus=modulus, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss = None
    for _step in range(steps + 1):
        optimizer.zero_grad()
        loss = sum(
            torch.nn.functional.cross_entropy(model(x, route_mask=mask), y)
            for mask in ((1.0, 1.0), (1.0, 0.0), (0.0, 1.0))
        )
        loss.backward()
        optimizer.step()

    metrics = evaluate_shared_residual_route_interventions(model, modulus=modulus, device=device)
    assert loss is not None
    return model, TinySharedResidualTrainingResult(
        final_loss=float(loss.detach().cpu()),
        base_accuracy=float(metrics["base_accuracy"]),
        route_a_only_accuracy=float(metrics["route_a_only_accuracy"]),
        route_b_only_accuracy=float(metrics["route_b_only_accuracy"]),
    )


def train_tiny_shared_residual_single_route_control(
    modulus: int = 7,
    steps: int = 160,
    lr: float = 3e-3,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[TinySharedResidualRedundantTransformer, TinySharedResidualTrainingResult]:
    """Train a shared residual model where the second route is decorative."""

    torch = _require_torch()
    torch.manual_seed(seed)
    model = TinySharedResidualRedundantTransformer(modulus=modulus).to(device)
    with torch.no_grad():
        model.route_b_output.weight.zero_()
        model.route_b_output.bias.zero_()
    for parameter in model.route_b_output.parameters():
        parameter.requires_grad = False

    x, y = modular_addition_dataset(modulus=modulus, device=device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=lr,
        weight_decay=1e-4,
    )
    loss = None
    for _step in range(steps + 1):
        optimizer.zero_grad()
        loss = sum(
            torch.nn.functional.cross_entropy(model(x, route_mask=mask), y)
            for mask in ((1.0, 1.0), (1.0, 0.0))
        )
        loss.backward()
        optimizer.step()

    metrics = evaluate_shared_residual_route_interventions(model, modulus=modulus, device=device)
    assert loss is not None
    return model, TinySharedResidualTrainingResult(
        final_loss=float(loss.detach().cpu()),
        base_accuracy=float(metrics["base_accuracy"]),
        route_a_only_accuracy=float(metrics["route_a_only_accuracy"]),
        route_b_only_accuracy=float(metrics["route_b_only_accuracy"]),
    )


def _accuracy(
    model: TinySharedResidualRedundantTransformer,
    tokens: Any,
    target: Any,
    route_mask: tuple[float, float],
) -> float:
    with _require_torch().no_grad():
        pred = model(tokens, route_mask=route_mask).argmax(dim=-1)
        return float((pred == target).float().mean().detach().cpu())


def evaluate_shared_residual_route_interventions(
    model: TinySharedResidualRedundantTransformer,
    modulus: int = 7,
    device: str = "cpu",
    single_drop_threshold: float = 0.1,
    dual_drop_threshold: float = 0.5,
) -> dict[str, float | bool]:
    model.eval()
    x, y = modular_addition_dataset(modulus=modulus, device=device)
    base = _accuracy(model, x, y, (1.0, 1.0))
    route_a_only = _accuracy(model, x, y, (1.0, 0.0))
    route_b_only = _accuracy(model, x, y, (0.0, 1.0))
    both_ablate = _accuracy(model, x, y, (0.0, 0.0))
    drop_ablate_a = base - route_b_only
    drop_ablate_b = base - route_a_only
    dual_drop = base - both_ablate
    max_single_drop = max(drop_ablate_a, drop_ablate_b)
    return {
        "base_accuracy": base,
        "route_a_only_accuracy": route_a_only,
        "route_b_only_accuracy": route_b_only,
        "both_routes_ablated_accuracy": both_ablate,
        "drop_when_ablate_route_a": drop_ablate_a,
        "drop_when_ablate_route_b": drop_ablate_b,
        "max_single_ablation_drop": max_single_drop,
        "dual_ablation_drop": dual_drop,
        "single_ablation_underweights": bool(max_single_drop <= single_drop_threshold and dual_drop >= dual_drop_threshold),
        "redundancy_certified": bool(
            route_a_only >= base - single_drop_threshold
            and route_b_only >= base - single_drop_threshold
            and dual_drop >= dual_drop_threshold
        ),
    }


def run_shared_residual_redundant_cpu_sweep(
    seeds: Sequence[int] = (0, 1, 2),
    modulus: int = 7,
    steps: int = 160,
) -> dict[str, object]:
    if not torch_available():
        return {"available": False, "reason": "optional torch dependency is not installed"}

    rows = []
    accounting: dict[str, int | float] | None = None
    for seed in seeds:
        model, training = train_tiny_shared_residual_redundant_transformer(
            modulus=modulus,
            steps=steps,
            seed=seed,
        )
        if accounting is None:
            accounting = model.parameter_accounting()
        interventions = evaluate_shared_residual_route_interventions(model, modulus=modulus)
        rows.append({"seed": seed, "training": training.as_dict(), "interventions": interventions})

    def mean(key: str) -> float:
        return float(sum(float(row["interventions"][key]) for row in rows) / len(rows))

    return {
        "available": True,
        "modulus": modulus,
        "steps": steps,
        "n_seeds": len(rows),
        "parameter_accounting": accounting or {},
        "rows": rows,
        "mean_base_accuracy": mean("base_accuracy"),
        "mean_route_a_only_accuracy": mean("route_a_only_accuracy"),
        "mean_route_b_only_accuracy": mean("route_b_only_accuracy"),
        "mean_both_routes_ablated_accuracy": mean("both_routes_ablated_accuracy"),
        "mean_max_single_ablation_drop": mean("max_single_ablation_drop"),
        "mean_dual_ablation_drop": mean("dual_ablation_drop"),
        "redundancy_certified_rate": float(
            sum(bool(row["interventions"]["redundancy_certified"]) for row in rows) / len(rows)
        ),
        "single_ablation_underweights_rate": float(
            sum(bool(row["interventions"]["single_ablation_underweights"]) for row in rows) / len(rows)
        ),
    }


def run_shared_residual_single_route_control_sweep(
    seeds: Sequence[int] = (0, 1, 2),
    modulus: int = 7,
    steps: int = 160,
) -> dict[str, object]:
    if not torch_available():
        return {"available": False, "reason": "optional torch dependency is not installed"}

    rows = []
    accounting: dict[str, int | float] | None = None
    for seed in seeds:
        model, training = train_tiny_shared_residual_single_route_control(
            modulus=modulus,
            steps=steps,
            seed=seed,
        )
        if accounting is None:
            accounting = model.parameter_accounting()
        interventions = evaluate_shared_residual_route_interventions(model, modulus=modulus)
        rows.append({"seed": seed, "training": training.as_dict(), "interventions": interventions})

    def mean(key: str) -> float:
        return float(sum(float(row["interventions"][key]) for row in rows) / len(rows))

    return {
        "available": True,
        "modulus": modulus,
        "steps": steps,
        "n_seeds": len(rows),
        "parameter_accounting": accounting or {},
        "rows": rows,
        "mean_base_accuracy": mean("base_accuracy"),
        "mean_route_a_only_accuracy": mean("route_a_only_accuracy"),
        "mean_route_b_only_accuracy": mean("route_b_only_accuracy"),
        "mean_both_routes_ablated_accuracy": mean("both_routes_ablated_accuracy"),
        "mean_max_single_ablation_drop": mean("max_single_ablation_drop"),
        "mean_dual_ablation_drop": mean("dual_ablation_drop"),
        "redundancy_certified_rate": float(
            sum(bool(row["interventions"]["redundancy_certified"]) for row in rows) / len(rows)
        ),
        "single_ablation_underweights_rate": float(
            sum(bool(row["interventions"]["single_ablation_underweights"]) for row in rows) / len(rows)
        ),
        "false_redundancy_rate": float(
            sum(bool(row["interventions"]["redundancy_certified"]) for row in rows) / len(rows)
        ),
    }
