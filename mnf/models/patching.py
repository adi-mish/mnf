from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from mnf.models.activation_cache import collect_activations


def _first_tensor(value: Any) -> Any:
    if hasattr(value, "clone"):
        return value
    if isinstance(value, tuple):
        for item in value:
            if hasattr(item, "clone"):
                return item
    raise TypeError("module output does not contain a patchable tensor")


def _replace_first_tensor(original: Any, tensor: Any) -> Any:
    if hasattr(original, "clone"):
        return tensor
    if isinstance(original, tuple):
        replaced = []
        used = False
        for item in original:
            if not used and hasattr(item, "clone"):
                replaced.append(tensor)
                used = True
            else:
                replaced.append(item)
        return tuple(replaced)
    raise TypeError("module output does not contain a patchable tensor")


def run_with_token_activation_patch(
    model: Any,
    base_tokens: Any,
    source_tokens: Any,
    module_name: str = "embedding",
    token_positions: Sequence[int] = (0,),
) -> Any:
    _, source_cache = collect_activations(model, source_tokens, (module_name,))
    source_activation = source_cache[module_name]
    source_tensor = _first_tensor(source_activation)
    modules = dict(model.named_modules())
    if module_name not in modules:
        raise ValueError(f"unknown module: {module_name}")

    def patch_hook(_module: Any, _inputs: Any, output: Any) -> Any:
        output_tensor = _first_tensor(output)
        patched = output_tensor.clone()
        for position in token_positions:
            patched[:, position, :] = source_tensor[:, position, :]
        return _replace_first_tensor(output, patched)

    handle = modules[module_name].register_forward_hook(patch_hook)
    try:
        return model(base_tokens)
    finally:
        handle.remove()
