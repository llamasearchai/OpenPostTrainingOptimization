from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from rich.console import Console
from rich.table import Table

from ..utils.hardware_utils import DeviceInfo, dev_synchronize, ensure_dir, to_device

console = Console()

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


@dataclass
class QuantizationResult:
    backend: str
    method: str
    original_size_mb: Optional[float]
    quantized_size_mb: Optional[float]
    speed_toks_per_s: Optional[float]
    speedup: Optional[float]
    output_dir: str


def _param_count(model: Any) -> Optional[int]:
    try:
        import torch as _torch  # local import in case of missing torch

        if isinstance(model, _torch.nn.Module):
            return sum(p.numel() for p in model.parameters())
    except Exception:
        pass
    try:
        # MLX model may expose .parameters() too; otherwise skip
        return sum(getattr(p, "size", lambda: 0)() for p in model.parameters())  # type: ignore
    except Exception:
        return None


def _dtype_bytes_default(backend: str) -> int:
    # Assume fp16 for accel backends, fp32 for CPU
    return 2 if backend in {"mps", "mlx", "cuda"} else 4


def _method_to_bits(method: str) -> int:
    m = method.lower()
    if "4" in m:
        return 4
    if "8" in m:
        return 8
    return 8


def _print_results(result: QuantizationResult, model_name: str) -> None:
    table = Table(title="Quantization Results")
    table.add_column("Metric")
    table.add_column("Before")
    table.add_column("After")
    table.add_column("Improvement")

    before = f"{result.original_size_mb:.1f} MB" if result.original_size_mb else "-"
    after = f"{result.quantized_size_mb:.1f} MB" if result.quantized_size_mb else "-"
    imp = (
        f"{(1 - (result.quantized_size_mb or 0) / (result.original_size_mb or 1)) * 100:.1f}%"
        if (result.original_size_mb and result.quantized_size_mb)
        else "-"
    )
    table.add_row("Model Size", before, after, imp)

    spd = f"{result.speed_toks_per_s:.2f} tok/s" if result.speed_toks_per_s else "-"
    table.add_row("Inference Speed", "-", spd, f"{result.speedup or 1.0:.2f}x")
    console.print(table)


def _save_config(result: QuantizationResult, output_dir: str) -> None:
    payload: Dict[str, Any] = {
        "method": result.method,
        "backend": result.backend,
        "metrics": {
            "original_size_mb": result.original_size_mb,
            "quantized_size_mb": result.quantized_size_mb,
            "speedup": result.speedup,
            "tokens_per_second": result.speed_toks_per_s,
        },
    }
    ensure_dir(output_dir)
    with open(Path(output_dir) / "quantization_config.json", "w") as f:
        json.dump(payload, f)


def _dir_size_mb(path: Path) -> Optional[float]:
    try:
        total = 0
        for p in path.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
        return total / (1024**2)
    except Exception:
        return None


def _mlx_quantize_load(model_name: str, method: str) -> Tuple[Any, Any]:
    from mlx_lm import load

    q = "q4" if "4" in method else "q8"
    model, tokenizer = load(model_name, quantize=q)
    return model, tokenizer


def _torch_quanto_load(model_name: str, method: str, device: DeviceInfo) -> Tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        from optimum.quanto import QuantoConfig
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Optimum-Quanto is required for int8/int4 on CPU/MPS. Install with [apple]."
        ) from e

    weights = "int4" if "4" in method else "int8"
    qconfig = QuantoConfig(weights=weights)  # type: ignore[arg-type]

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=qconfig,  # type: ignore[arg-type]
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = to_device(model, device)
    return model, tokenizer


def _bnb_cuda_load(model_name: str, method: str) -> Tuple[Any, Any]:  # pragma: no cover - CUDA
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_in_8bit = "8" in method
    load_in_4bit = "4" in method
    kwargs: Dict[str, Any] = {}
    if load_in_8bit:
        kwargs["load_in_8bit"] = True
    if load_in_4bit:
        kwargs["load_in_4bit"] = True
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def _generate_benchmark_tps(model: Any, tokenizer: Any, device: DeviceInfo) -> float:
    prompt = "Hello"
    max_new_tokens = 16
    if device.backend == "mlx":
        try:
            from mlx_lm import generate

            start = time.perf_counter()
            _ = generate(model, tokenizer, prompt, max_tokens=max_new_tokens)
            dev_synchronize(device.backend)
            end = time.perf_counter()
            return max_new_tokens / (end - start)
        except Exception:
            return 0.0
    if torch is None:
        return 0.0
    from transformers import TextIteratorStreamer
    import threading

    inputs = tokenizer(prompt, return_tensors="pt").to(device.device_str)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    kw = dict(max_new_tokens=max_new_tokens, do_sample=False)
    start = time.perf_counter()
    thread = threading.Thread(target=model.generate, kwargs={**inputs, "streamer": streamer, **kw})
    thread.start()
    for _ in streamer:
        pass
    thread.join()
    dev_synchronize(device.backend)
    end = time.perf_counter()
    return max_new_tokens / (end - start)


def quantize_model(
    model_name: str,
    method: str,
    output_dir: str,
    device: DeviceInfo,
) -> QuantizationResult:
    ensure_dir(output_dir)

    if device.backend == "mlx":
        if "gptq" in method.lower() or "awq" in method.lower():
            raise RuntimeError("Method 'gptq' requires CUDA.")
        model, tokenizer = _mlx_quantize_load(model_name, method)
        pc = None  # MLX params not easily counted in a backend-agnostic way
        orig_size = None
        bits = _method_to_bits(method)
        # Approximate size from HF config if available
        try:
            from transformers import AutoConfig

            cfg = AutoConfig.from_pretrained(model_name)
            hidden = int(getattr(cfg, "n_embd", getattr(cfg, "hidden_size", 0)))
            vocab = int(getattr(cfg, "vocab_size", 0))
            layers = int(getattr(cfg, "n_layer", getattr(cfg, "num_hidden_layers", 0)))
            approx_params = vocab * hidden + layers * (hidden * hidden * 12)
            pc = approx_params
        except Exception:
            pc = None
        if pc is not None:
            orig_size = pc * _dtype_bytes_default(device.backend) / (1024**2)
        # Try to persist MLX model to measure quantized size
        outp = Path(output_dir)
        ensure_dir(str(outp))
        q_size = None
        try:
            import mlx_lm as _mlxlm  # type: ignore

            if hasattr(_mlxlm, "save"):
                # Some versions support save(model, tokenizer, path)
                try:
                    _mlxlm.save(model, tokenizer, str(outp))  # type: ignore[attr-defined]
                    q_size = _dir_size_mb(outp)
                except Exception:
                    pass
            else:
                # Try utils.save
                try:
                    from mlx_lm.utils import save as _mlx_save  # type: ignore

                    _mlx_save(model, tokenizer, str(outp))
                    q_size = _dir_size_mb(outp)
                except Exception:
                    pass
        except Exception:
            pass
        # Fallback to approximate quantized size if save was not possible
        if q_size is None and pc is not None:
            q_size = pc * (bits / 8) / (1024**2)
        tps = _generate_benchmark_tps(model, tokenizer, device)
        res = QuantizationResult(
            backend=device.backend,
            method="q4" if bits == 4 else "q8",
            original_size_mb=orig_size,
            quantized_size_mb=q_size,
            speed_toks_per_s=tps if tps > 0 else None,
            speedup=None,
            output_dir=output_dir,
        )
        _save_config(res, output_dir)
        # Save minimal MLX metadata
        meta = {
            "model": model_name,
            "quantize": "q4" if bits == 4 else "q8",
            "backend": device.backend,
        }
        with open(Path(output_dir) / "mlx_metadata.json", "w") as f:
            json.dump(meta, f)
        _print_results(res, model_name)
        return res

    if device.backend in {"mps", "cpu"}:
        if method.lower() in {"gptq", "awq"}:
            raise RuntimeError("Method 'gptq' requires CUDA.")
        model, tokenizer = _torch_quanto_load(model_name, method, device)
        # Save HF artifacts and measure quantized size
        outp = Path(output_dir)
        ensure_dir(str(outp))
        model.save_pretrained(outp)
        try:
            tokenizer.save_pretrained(outp)
        except Exception:
            pass
        q_size = _dir_size_mb(outp)
        # Estimate original size from param count and assumed fp16 baseline
        pc = _param_count(model)
        orig_size = (pc * _dtype_bytes_default(device.backend) / (1024**2)) if pc else None
        bits = _method_to_bits(method)
        tps = _generate_benchmark_tps(model, tokenizer, device)
        res = QuantizationResult(
            backend=device.backend,
            method=method.lower(),
            original_size_mb=orig_size,
            quantized_size_mb=q_size,
            speed_toks_per_s=tps if tps > 0 else None,
            speedup=None,
            output_dir=output_dir,
        )
        _save_config(res, output_dir)
        _print_results(res, model_name)
        return res

    # CUDA path
    if device.backend == "cuda":  # pragma: no cover - exercised on CUDA hosts
        if method.lower() in {"int8", "int4"}:
            model, tokenizer = _bnb_cuda_load(model_name, method)
        else:
            raise RuntimeError(f"Unsupported method '{method}' on CUDA in this simplified build.")
        pc = _param_count(model)
        orig_size = (pc * _dtype_bytes_default(device.backend) / (1024**2)) if pc else None
        bits = _method_to_bits(method)
        q_size = (pc * (bits / 8) / (1024**2)) if pc else None
        tps = _generate_benchmark_tps(model, tokenizer, device)
        res = QuantizationResult(
            backend=device.backend,
            method=method.lower(),
            original_size_mb=orig_size,
            quantized_size_mb=q_size,
            speed_toks_per_s=tps if tps > 0 else None,
            speedup=None,
            output_dir=output_dir,
        )
        _save_config(res, output_dir)
        _print_results(res, model_name)
        return res

    raise RuntimeError(f"Unknown backend: {device.backend}")
