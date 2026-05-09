from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

try:
    import torch as _torch
except ImportError:
    _torch = None


def torch_available() -> bool:
    return _torch is not None


def _require_torch() -> Any:
    if _torch is None:
        raise ImportError("tiny transformer benchmarks require the optional torch dependency")
    return _torch


def modular_addition_dataset(modulus: int = 7, device: str = "cpu") -> tuple[Any, Any]:
    torch = _require_torch()
    xs = []
    ys = []
    for a in range(modulus):
        for b in range(modulus):
            xs.append([a, b, modulus])
            ys.append((a + b) % modulus)
    return torch.tensor(xs, dtype=torch.long, device=device), torch.tensor(ys, dtype=torch.long, device=device)


_BaseModule = _torch.nn.Module if _torch is not None else object


class TinyModularAdditionTransformer(_BaseModule):
    def __init__(
        self,
        modulus: int = 7,
        d_model: int = 32,
        n_heads: int = 4,
        d_ff: int = 64,
        n_layers: int = 1,
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
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=n_layers)
        self.output = torch.nn.Linear(d_model, modulus)

    def forward(self, tokens: Any) -> Any:
        hidden = self.embedding(tokens) + self.position.unsqueeze(0)
        encoded = self.encoder(hidden)
        return self.output(encoded[:, -1])


@dataclass(frozen=True)
class TinyTrainingResult:
    train_loss: tuple[float, ...]
    train_accuracy: tuple[float, ...]
    final_accuracy: float
    final_loss: float

    def as_dict(self) -> dict[str, object]:
        return {
            "train_loss": list(self.train_loss),
            "train_accuracy": list(self.train_accuracy),
            "final_accuracy": self.final_accuracy,
            "final_loss": self.final_loss,
        }


def train_tiny_modular_addition(
    modulus: int = 7,
    steps: int = 160,
    lr: float = 3e-3,
    seed: int = 0,
    d_model: int = 32,
    n_heads: int = 4,
    device: str = "cpu",
    log_every: int = 20,
) -> tuple[TinyModularAdditionTransformer, TinyTrainingResult]:
    torch = _require_torch()
    torch.manual_seed(seed)
    model = TinyModularAdditionTransformer(modulus=modulus, d_model=d_model, n_heads=n_heads).to(device)
    x, y = modular_addition_dataset(modulus=modulus, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    losses: list[float] = []
    accuracies: list[float] = []
    for step in range(steps + 1):
        optimizer.zero_grad()
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        if step % log_every == 0 or step == steps:
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                losses.append(float(loss.detach().cpu()))
                accuracies.append(float((pred == y).float().mean().detach().cpu()))

    with torch.no_grad():
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        acc = float((logits.argmax(dim=-1) == y).float().mean().detach().cpu())
    return model, TinyTrainingResult(
        train_loss=tuple(losses),
        train_accuracy=tuple(accuracies),
        final_accuracy=acc,
        final_loss=float(loss.detach().cpu()),
    )


def _accuracy(logits: Any, target: Any) -> float:
    return float((logits.argmax(dim=-1) == target).float().mean().detach().cpu())


def evaluate_modular_interventions(
    model: TinyModularAdditionTransformer,
    modulus: int = 7,
    deltas: Sequence[int] = (1, 2, 3),
    device: str = "cpu",
) -> dict[str, float]:
    torch = _require_torch()
    model.eval()
    x, y = modular_addition_dataset(modulus=modulus, device=device)
    with torch.no_grad():
        base_pred = model(x).argmax(dim=-1)
        base_accuracy = float((base_pred == y).float().mean().detach().cpu())
        a_effects = []
        b_effects = []
        a_shift_consistency = []
        b_shift_consistency = []
        for delta in deltas:
            a_shifted = x.clone()
            a_shifted[:, 0] = (a_shifted[:, 0] + delta) % modulus
            a_pred = model(a_shifted).argmax(dim=-1)
            a_effects.append(float((a_pred != base_pred).float().mean().detach().cpu()))
            a_expected = (base_pred + delta) % modulus
            a_shift_consistency.append(float((a_pred == a_expected).float().mean().detach().cpu()))

            b_shifted = x.clone()
            b_shifted[:, 1] = (b_shifted[:, 1] + delta) % modulus
            b_pred = model(b_shifted).argmax(dim=-1)
            b_effects.append(float((b_pred != base_pred).float().mean().detach().cpu()))
            b_expected = (base_pred + delta) % modulus
            b_shift_consistency.append(float((b_pred == b_expected).float().mean().detach().cpu()))

        a_zero = x.clone()
        a_zero[:, 0] = 0
        b_zero = x.clone()
        b_zero[:, 1] = 0
        a_zero_original_accuracy = _accuracy(model(a_zero), y)
        b_zero_original_accuracy = _accuracy(model(b_zero), y)

    return {
        "base_accuracy": base_accuracy,
        "a_shift_effect_rate": float(sum(a_effects) / len(a_effects)),
        "b_shift_effect_rate": float(sum(b_effects) / len(b_effects)),
        "a_cyclic_shift_consistency": float(sum(a_shift_consistency) / len(a_shift_consistency)),
        "b_cyclic_shift_consistency": float(sum(b_shift_consistency) / len(b_shift_consistency)),
        "a_zero_original_accuracy": a_zero_original_accuracy,
        "b_zero_original_accuracy": b_zero_original_accuracy,
    }
