from __future__ import annotations

from mnf.integrations.hf_transformers import run_tiny_causal_lm_smoke


def run(model_name: str = "sshleifer/tiny-gpt2", local_files_only: bool = True) -> dict[str, object]:
    return run_tiny_causal_lm_smoke(model_name=model_name, local_files_only=local_files_only).as_dict()
