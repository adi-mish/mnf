from __future__ import annotations

from mnf.integrations.hf_transformers import run_tiny_causal_lm_smoke
from mnf.integrations.tracr import run_tracr_smoke
from mnf.integrations.transformer_lens import run_random_hooked_transformer_smoke


def run(model_name: str = "sshleifer/tiny-gpt2", local_files_only: bool = True) -> dict[str, object]:
    hf = run_tiny_causal_lm_smoke(model_name=model_name, local_files_only=local_files_only).as_dict()
    tl = run_random_hooked_transformer_smoke().as_dict()
    out = dict(hf)
    out["transformer_lens_smoke"] = tl
    out["tracr_smoke"] = run_tracr_smoke()
    return out
