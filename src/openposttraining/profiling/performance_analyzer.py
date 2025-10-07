from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil

from ..deployment.inference_engine import InferenceEngine
from ..utils.hardware_utils import DeviceInfo, detect_device, dev_synchronize, ensure_dir


def _rss_gb() -> float:
    return psutil.Process().memory_info().rss / 1e9


def profile_latency(
    engine: InferenceEngine,
    prompt: str = "Hello",
    iters: int = 5,
    max_tokens: int = 16,
) -> Dict[str, float]:
    times: List[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        _ = engine.generate(prompt, max_tokens=max_tokens)
        dev_synchronize(engine.device.backend)
        end = time.perf_counter()
        times.append((end - start) * 1000.0)
    return {
        "mean": statistics.mean(times),
        "p95": statistics.quantiles(times, n=100)[94] if len(times) >= 2 else times[0],
    }


def profile_throughput(
    engine: InferenceEngine,
    prompt: str = "Hello",
    iters: int = 5,
    max_tokens: int = 32,
) -> Dict[str, float]:
    tokens_per_second: List[float] = []
    total_time = 0.0
    for _ in range(iters):
        start = time.perf_counter()
        _ = engine.generate(prompt, max_tokens=max_tokens)
        dev_synchronize(engine.device.backend)
        end = time.perf_counter()
        dt = end - start
        total_time += dt
        tokens_per_second.append(max_tokens / dt)
    sps = iters / total_time if total_time > 0 else 0.0
    return {"tokens_per_second": statistics.mean(tokens_per_second), "samples_per_second": sps}


def profile_memory(engine: InferenceEngine) -> Dict[str, float]:
    # Encourage memory cleanup on MPS before measuring
    try:
        import torch

        if engine.device.backend == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()  # type: ignore[attr-defined]
    except Exception:
        pass
    before = _rss_gb()
    _ = engine.generate("Hello", max_tokens=8)
    dev_synchronize(engine.device.backend)
    try:
        import torch

        if engine.device.backend == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()  # type: ignore[attr-defined]
    except Exception:
        pass
    after = _rss_gb()
    return {"peak_gb": max(before, after), "model_gb": after - before if after > before else 0.0, "activation_gb": 0.0, "reserved_gb": 0.0}


def profile_entry(
    model_name: str,
    metrics: List[str],
    device: Optional[str],
    output_path: Optional[str] = None,
    batch_size: int = 1,
    iters: int = 5,
    input_len: int = 32,
) -> Dict[str, Any]:
    dev = detect_device(device)
    engine = InferenceEngine(model_name, prefer_device=dev.backend)
    engine.load_model()
    result: Dict[str, Any] = {}
    prompt = "Hello " + ("x" * max(0, input_len - 6))
    if "latency" in metrics:
        result["latency"] = profile_latency(engine, prompt=prompt, iters=iters)
    if "throughput" in metrics:
        result["throughput"] = profile_throughput(engine, prompt=prompt, iters=iters)
    if "memory" in metrics:
        result["memory"] = profile_memory(engine)
    if output_path:
        ensure_dir(str(Path(output_path).parent))
        with open(output_path, "w") as f:
            json.dump(result, f)
    return result
