from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from mnf.models.activation_cache import collect_activations


def run_with_token_activation_patch(
    model: Any,
    base_tokens: Any,
    source_tokens: Any,
    module_name: str = "embedding",
    token_positions: Sequence[int] = (0,),
) -> Any:
    _, source_cache = collect_activations(model, source_tokens, (module_name,))
    source_activation = source_cache[module_name]
    modules = dict(model.named_modules())
    if module_name not in modules:
        raise ValueError(f"unknown module: {module_name}")

    def patch_hook(_module: Any, _inputs: Any, output: Any) -> Any:
        patched = output.clone()
        for position in token_positions:
            patched[:, position, :] = source_activation[:, position, :]
        return patched

    handle = modules[module_name].register_forward_hook(patch_hook)
    try:
        return model(base_tokens)
    finally:
        handle.remove()
