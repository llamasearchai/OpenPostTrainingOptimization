from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Tuple

from ..utils.hardware_utils import DeviceInfo, ensure_dir, to_device

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def _load_hf_pair(main_name: str, draft_name: str, device: DeviceInfo):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok_main = AutoTokenizer.from_pretrained(main_name)
    tok_draft = AutoTokenizer.from_pretrained(draft_name)
    model_main = AutoModelForCausalLM.from_pretrained(main_name)
    model_draft = AutoModelForCausalLM.from_pretrained(draft_name)
    model_main = to_device(model_main, device)
    model_draft = to_device(model_draft, device)
    return (model_main, tok_main), (model_draft, tok_draft)


def speculative_entry(
    main_model: str,
    draft_model: str,
    gamma: int,
    device: DeviceInfo,
    benchmark: bool,
    output_dir: str,
) -> Dict[str, Any]:
    ensure_dir(output_dir)
    if device.backend == "mlx":
        from mlx_lm import load, generate

        m, t = load(main_model)
        d, td = load(draft_model)
        prompt = "Hello"
        start = time.perf_counter()
        # Simple approach: run draft to get draft_tokens; verify with main greedily
        draft_text = generate(d, td, prompt, max_tokens=gamma)
        main_text = generate(m, t, prompt, max_tokens=gamma)
        end = time.perf_counter()
        accept = sum(1 for a, b in zip(draft_text, main_text) if a == b)
        acc_rate = accept / max(1, len(draft_text))
        tps = gamma / (end - start)
        metrics = {"tokens_per_second": tps, "acceptance_rate": acc_rate, "latency_ms": (end - start) * 1000}
    else:
        (m, tm), (d, td) = _load_hf_pair(main_model, draft_model, device)
        prompt = "Hello"
        inputs_m = tm(prompt, return_tensors="pt").to(device.device_str)
        inputs_d = td(prompt, return_tensors="pt").to(device.device_str)
        start = time.perf_counter()
        with torch.no_grad():  # type: ignore[union-attr]
            out_d = d.generate(**inputs_d, max_new_tokens=gamma, do_sample=False)
            out_m = m.generate(**inputs_m, max_new_tokens=gamma, do_sample=False)
        end = time.perf_counter()
        tdraft = td.decode(out_d[0], skip_special_tokens=True)
        tmain = tm.decode(out_m[0], skip_special_tokens=True)
        accept = sum(1 for a, b in zip(tdraft, tmain) if a == b)
        acc_rate = accept / max(1, len(tdraft))
        tps = gamma / (end - start)
        metrics = {"tokens_per_second": tps, "acceptance_rate": acc_rate, "latency_ms": (end - start) * 1000}

    with open(Path(output_dir) / "speculative_metrics.json", "w") as f:
        json.dump(metrics, f)
    return metrics
