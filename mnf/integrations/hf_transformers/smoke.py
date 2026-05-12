from __future__ import annotations

from dataclasses import dataclass

from mnf.integrations.availability import DependencyStatus, check_optional_dependency


def hf_transformers_status() -> DependencyStatus:
    return check_optional_dependency("transformers", "transformers")


@dataclass(frozen=True)
class TinyCausalLMSmokeResult:
    available: bool
    model_name: str
    prompt: str
    logits_shape: tuple[int, ...] = ()
    n_hidden_states: int = 0
    final_hidden_shape: tuple[int, ...] = ()
    reason: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "available": self.available,
            "model_name": self.model_name,
            "prompt": self.prompt,
            "logits_shape": list(self.logits_shape),
            "n_hidden_states": self.n_hidden_states,
            "final_hidden_shape": list(self.final_hidden_shape),
            "reason": self.reason,
        }


def run_tiny_causal_lm_smoke(
    model_name: str = "sshleifer/tiny-gpt2",
    prompt: str = "The answer is",
    local_files_only: bool = False,
) -> TinyCausalLMSmokeResult:
    status = hf_transformers_status()
    if not status.available:
        return TinyCausalLMSmokeResult(False, model_name=model_name, prompt=prompt, reason=status.reason)
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        return TinyCausalLMSmokeResult(False, model_name=model_name, prompt=prompt, reason=repr(exc))
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=local_files_only)
        inputs = tokenizer(prompt, return_tensors="pt")
        output = model(**inputs, output_hidden_states=True)
    except Exception as exc:
        return TinyCausalLMSmokeResult(False, model_name=model_name, prompt=prompt, reason=repr(exc))
    hidden_states = tuple(output.hidden_states or ())
    final_hidden_shape = tuple(hidden_states[-1].shape) if hidden_states else ()
    return TinyCausalLMSmokeResult(
        True,
        model_name=model_name,
        prompt=prompt,
        logits_shape=tuple(output.logits.shape),
        n_hidden_states=len(hidden_states),
        final_hidden_shape=final_hidden_shape,
        reason="ok",
    )
