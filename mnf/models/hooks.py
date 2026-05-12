from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mnf.models.activation_cache import collect_activations
from mnf.models.tiny_transformer import TinyModularAdditionTransformer, _require_torch


@dataclass(frozen=True)
class TinyHookSpec:
    name: str
    semantic: str
    available: bool

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "semantic": self.semantic, "available": self.available}


def tiny_hook_specs(model: TinyModularAdditionTransformer) -> tuple[TinyHookSpec, ...]:
    modules = dict(model.named_modules())
    requested = (
        ("embedding", "token embedding output"),
        ("position", "positional stream parameter"),
        ("encoder.layers.0.self_attn.q", "attention query projection"),
        ("encoder.layers.0.self_attn.k", "attention key projection"),
        ("encoder.layers.0.self_attn.v", "attention value projection"),
        ("encoder.layers.0.self_attn.pattern", "attention pattern"),
        ("encoder.layers.0.self_attn.out", "attention output"),
        ("encoder.layers.0.linear1", "MLP preactivation"),
        ("encoder.layers.0.linear2", "MLP output"),
        ("encoder.layers.0.norm1", "residual stream after attention block"),
        ("encoder.layers.0.norm2", "residual stream after MLP block"),
        ("output", "logit readout"),
    )
    available = set(modules)
    derived = {
        "position",
        "encoder.layers.0.self_attn.q",
        "encoder.layers.0.self_attn.k",
        "encoder.layers.0.self_attn.v",
        "encoder.layers.0.self_attn.pattern",
        "encoder.layers.0.self_attn.out",
    }
    return tuple(TinyHookSpec(name, semantic, name in available or name in derived) for name, semantic in requested)


def _first_tensor(value: Any) -> Any:
    if hasattr(value, "shape"):
        return value
    if isinstance(value, tuple):
        for item in value:
            if hasattr(item, "shape"):
                return item
    raise TypeError("expected tensor-like activation")


def _attention_qkv_and_pattern(model: TinyModularAdditionTransformer, tokens: Any) -> dict[str, Any]:
    torch = _require_torch()
    layer = model.encoder.layers[0]
    attn = layer.self_attn
    hidden = model.embedding(tokens) + model.position.unsqueeze(0)
    weight = attn.in_proj_weight
    bias = attn.in_proj_bias
    q_weight, k_weight, v_weight = weight.chunk(3, dim=0)
    q_bias, k_bias, v_bias = bias.chunk(3, dim=0) if bias is not None else (None, None, None)
    q = torch.nn.functional.linear(hidden, q_weight, q_bias)
    k = torch.nn.functional.linear(hidden, k_weight, k_bias)
    v = torch.nn.functional.linear(hidden, v_weight, v_bias)
    batch, seq, d_model = q.shape
    n_heads = attn.num_heads
    head_dim = d_model // n_heads

    def split_heads(x: Any) -> Any:
        return x.reshape(batch, seq, n_heads, head_dim).transpose(1, 2)

    qh = split_heads(q)
    kh = split_heads(k)
    scores = torch.matmul(qh, kh.transpose(-2, -1)) / (head_dim**0.5)
    pattern = torch.softmax(scores, dim=-1)
    return {
        "encoder.layers.0.self_attn.q": q.detach().clone(),
        "encoder.layers.0.self_attn.k": k.detach().clone(),
        "encoder.layers.0.self_attn.v": v.detach().clone(),
        "encoder.layers.0.self_attn.pattern": pattern.detach().clone(),
    }


def collect_tiny_hook_activations(
    model: TinyModularAdditionTransformer,
    tokens: Any,
) -> dict[str, Any]:
    module_names = (
        "embedding",
        "encoder.layers.0.self_attn",
        "encoder.layers.0.linear1",
        "encoder.layers.0.linear2",
        "encoder.layers.0.norm1",
        "encoder.layers.0.norm2",
        "output",
    )
    with _require_torch().no_grad():
        _, cache = collect_activations(model, tokens, module_names)
        out = {
            "embedding": cache["embedding"],
            "position": model.position.detach().clone(),
            "encoder.layers.0.self_attn.out": _first_tensor(cache["encoder.layers.0.self_attn"]).detach().clone(),
            "encoder.layers.0.linear1": cache["encoder.layers.0.linear1"],
            "encoder.layers.0.linear2": cache["encoder.layers.0.linear2"],
            "encoder.layers.0.norm1": cache["encoder.layers.0.norm1"],
            "encoder.layers.0.norm2": cache["encoder.layers.0.norm2"],
            "output": cache["output"],
        }
        out.update(_attention_qkv_and_pattern(model, tokens))
    return out
