from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TracrHookRequest:
    site: str
    semantic: str

    def as_dict(self) -> dict[str, str]:
        return {"site": self.site, "semantic": self.semantic}


def default_tracr_hook_requests() -> tuple[TracrHookRequest, ...]:
    return (
        TracrHookRequest("residual_stream", "compiled program state"),
        TracrHookRequest("attention_pattern", "selector-like RASP operation"),
        TracrHookRequest("mlp_output", "compiled transform output"),
        TracrHookRequest("logits", "program output interface"),
    )
