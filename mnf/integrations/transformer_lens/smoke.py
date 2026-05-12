from __future__ import annotations

from dataclasses import dataclass

from mnf.integrations.transformer_lens.availability import transformer_lens_status


@dataclass(frozen=True)
class TransformerLensSmokeResult:
    available: bool
    logits_shape: tuple[int, ...] = ()
    n_cache_entries: int = 0
    cache_keys: tuple[str, ...] = ()
    reason: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "available": self.available,
            "logits_shape": list(self.logits_shape),
            "n_cache_entries": self.n_cache_entries,
            "cache_keys": list(self.cache_keys),
            "reason": self.reason,
        }


def run_random_hooked_transformer_smoke() -> TransformerLensSmokeResult:
    status = transformer_lens_status()
    if not status.available:
        return TransformerLensSmokeResult(False, reason=status.reason)
    try:
        import torch
        from transformer_lens import HookedTransformer, HookedTransformerConfig

        cfg = HookedTransformerConfig(
            n_layers=1,
            d_model=16,
            n_ctx=4,
            d_head=8,
            n_heads=2,
            d_mlp=32,
            d_vocab=32,
            act_fn="relu",
        )
        model = HookedTransformer(cfg)
        tokens = torch.tensor([[1, 2, 3, 4]])
        logits, cache = model.run_with_cache(tokens)
        keys = tuple(str(key) for key in cache.keys())
        return TransformerLensSmokeResult(
            True,
            logits_shape=tuple(logits.shape),
            n_cache_entries=len(keys),
            cache_keys=keys[:8],
            reason="ok",
        )
    except Exception as exc:
        return TransformerLensSmokeResult(False, reason=repr(exc))
