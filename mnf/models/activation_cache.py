from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ActivationCache:
    activations: dict[str, Any]

    def __getitem__(self, name: str) -> Any:
        return self.activations[name]

    def keys(self) -> tuple[str, ...]:
        return tuple(self.activations)


def clone_activation(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(clone_activation(item) if item is not None else None for item in value)
    if isinstance(value, list):
        return [clone_activation(item) if item is not None else None for item in value]
    return value


def collect_activations(model: Any, tokens: Any, module_names: Sequence[str]) -> tuple[Any, ActivationCache]:
    modules = dict(model.named_modules())
    missing = [name for name in module_names if name not in modules]
    if missing:
        raise ValueError(f"unknown modules: {missing}")

    activations: dict[str, Any] = {}
    handles = []

    def hook_for(name: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            activations[name] = clone_activation(output)

        return hook

    for name in module_names:
        handles.append(modules[name].register_forward_hook(hook_for(name)))
    try:
        output = model(tokens)
    finally:
        for handle in handles:
            handle.remove()
    return output, ActivationCache(activations)
