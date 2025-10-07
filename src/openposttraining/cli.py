from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.table import Table

from . import __version__
from .core.quantization import quantize_model
from .core.sparsity import sparsify_entry
from .core.speculative import speculative_entry
from .deployment.inference_engine import InferenceEngine
from .profiling.performance_analyzer import profile_entry
from .utils.hardware_utils import device_banner, detect_device, ensure_dir, estimate_model_mem_gb
from .models.llm_optimizer import LLMOptimizer

console = Console()


def cmd_status(args: argparse.Namespace) -> int:
    dev = detect_device(args.device)
    model_mem = estimate_model_mem_gb(args.model, dev.backend)
    console.print(device_banner(__version__, dev, model_name=args.model or "-", model_mem_gb=model_mem))
    table = Table(title="Device Status")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Backend", dev.backend)
    table.add_row("Device", dev.device_str)
    table.add_row("RSS", f"{dev.rss_gb:.2f} GB")
    table.add_row("RAM", f"{dev.total_ram_gb:.1f} GB")
    table.add_row("Flash-Attn", "yes" if dev.supports_flash_attn else "no")
    table.add_row("bitsandbytes", "yes" if dev.supports_bnb else "no")
    console.print(table)
    if args.export:
        ensure_dir(str(Path(args.export).parent))
        with open(args.export, "w") as f:
            json.dump(
                {
                    "backend": dev.backend,
                    "device": dev.device_str,
                    "rss_gb": dev.rss_gb,
                    "total_ram_gb": dev.total_ram_gb,
                    "supports_flash_attn": dev.supports_flash_attn,
                    "supports_bnb": dev.supports_bnb,
                },
                f,
            )
    return 0


def cmd_quantize(args: argparse.Namespace) -> int:
    dev = detect_device(args.device)
    model_mem = estimate_model_mem_gb(args.model, dev.backend)
    console.print(device_banner(__version__, dev, model_name=args.model, model_mem_gb=model_mem))
    try:
        _ = quantize_model(args.model, args.method, args.output, dev)
        return 0
    except Exception as e:
        console.print(f"[red]Quantization failed:[/red] {e}")
        return 1


def cmd_sparsify(args: argparse.Namespace) -> int:
    dev = detect_device(args.device)
    model_mem = estimate_model_mem_gb(args.model, dev.backend)
    console.print(device_banner(__version__, dev, model_name=args.model, model_mem_gb=model_mem))
    try:
        metrics = sparsify_entry(args.model, sparsity=args.sparsity, pattern=args.pattern, output_dir=args.output, device=dev)
    except Exception as e:
        console.print(f"[red]Sparsify failed:[/red] {e}")
        return 1
    table = Table(title="Sparsity Results")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Sparsity Applied", f"{metrics['actual_sparsity']*100:.1f}%")
    table.add_row("Theoretical Speedup", f">= {metrics['speedup_theoretical']:.1f}x")
    console.print(table)
    return 0


def cmd_profile(args: argparse.Namespace) -> int:
    dev = detect_device(args.device)
    model_mem = estimate_model_mem_gb(args.model, dev.backend)
    console.print(device_banner(__version__, dev, model_name=args.model, model_mem_gb=model_mem))
    metrics = [m.strip() for m in args.profile]
    try:
        result = profile_entry(
            args.model,
            metrics=metrics,
            device=dev.backend,
            output_path=args.export,
            batch_size=args.batch_size,
            iters=args.iters,
            input_len=args.input_len,
        )
    except Exception as e:
        console.print(f"[red]Profile failed:[/red] {e}")
        return 1
    # Pretty tables per metric
    if "latency" in result:
        t = Table(title="Latency")
        t.add_column("mean (ms)")
        t.add_column("p95 (ms)")
        lat = result["latency"]
        t.add_row(f"{lat.get('mean', 0):.2f}", f"{lat.get('p95', 0):.2f}")
        console.print(t)
    if "throughput" in result:
        t = Table(title="Throughput")
        t.add_column("tokens/sec")
        t.add_column("samples/sec")
        thr = result["throughput"]
        t.add_row(f"{thr.get('tokens_per_second', 0):.2f}", f"{thr.get('samples_per_second', 0):.2f}")
        console.print(t)
    if "memory" in result:
        t = Table(title="Memory")
        t.add_column("peak (GB)")
        t.add_column("model (GB)")
        t.add_column("activation (GB)")
        t.add_column("reserved (GB)")
        mem = result["memory"]
        t.add_row(
            f"{mem.get('peak_gb', 0):.2f}",
            f"{mem.get('model_gb', 0):.2f}",
            f"{mem.get('activation_gb', 0):.2f}",
            f"{mem.get('reserved_gb', 0):.2f}",
        )
        console.print(t)
    # Also print machine-readable JSON
    console.print_json(json.dumps(result))
    return 0


def cmd_speculative(args: argparse.Namespace) -> int:
    dev = detect_device(args.device)
    model_mem = estimate_model_mem_gb(args.model, dev.backend)
    console.print(device_banner(__version__, dev, model_name=args.model, model_mem_gb=model_mem))
    try:
        metrics = speculative_entry(
            main_model=args.model,
            draft_model=args.draft,
            gamma=args.gamma,
            device=dev,
            benchmark=args.benchmark,
            output_dir=args.output,
        )
    except Exception as e:
        console.print(f"[red]Speculative failed:[/red] {e}")
        return 1
    table = Table(title="Speculative Decoding")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Tokens/sec", f"{metrics['tokens_per_second']:.2f}")
    table.add_row("Acceptance Rate", f"{metrics['acceptance_rate']:.2f}")
    table.add_row("Latency (ms)", f"{metrics['latency_ms']:.2f}")
    console.print(table)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    dev = detect_device(args.device)
    model_mem = estimate_model_mem_gb(args.model, dev.backend)
    console.print(device_banner(__version__, dev, model_name=args.model, model_mem_gb=model_mem))
    backend = (args.backend or ("mlx" if dev.backend == "mlx" else "llama_cpp")).lower()
    if backend == "vllm":
        console.print("vLLM MPS not supported; use '--backend mlx' or 'llama_cpp'.")
        return 0
    if backend == "mlx":
        from .deployment.servers.mlx_server import create_app

        app = create_app(args.model)
    elif backend == "llama_cpp":
        from .deployment.servers.llamacpp_server import create_app

        app = create_app(args.model)
    else:
        console.print(f"Unknown backend: {backend}")
        return 1
    try:
        import uvicorn
    except Exception as e:
        console.print("Uvicorn is required. Install with extras 'serve'.")
        return 1
    console.log(f"Serving ({backend}) at http://0.0.0.0:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
    return 0


def _convert_coreml(args: argparse.Namespace) -> int:
    # Minimal demo conversion: export a tiny embedding-like module
    try:
        import torch
        import coremltools as ct
    except Exception as e:
        console.print("coremltools and torch are required for Core ML export.")
        return 1

    class TinyTok(torch.nn.Module):
        def __init__(self, vocab: int = 256, dim: int = 64):
            super().__init__()
            self.emb = torch.nn.Embedding(vocab, dim)

        def forward(self, ids):
            x = self.emb(ids)
            return x.mean(dim=1)

    model = TinyTok()
    model.eval()
    example = torch.randint(0, 255, (1, 8), dtype=torch.long)
    traced = torch.jit.trace(model, example)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="ids", shape=example.shape, dtype=ct.int64)],
    )
    out_dir = Path(args.output)
    ensure_dir(str(out_dir))
    mlmodel_path = out_dir / "model.mlmodel"
    mlmodel.save(mlmodel_path)
    runner = out_dir / "run_coreml.py"
    runner.write_text(
        """
import sys
import numpy as np
import coremltools as ct

mlmodel = ct.models.MLModel('model.mlmodel')

text = sys.argv[1] if len(sys.argv) > 1 else 'Hello'
# Map characters to [0,255] token ids and pad/truncate to length 8
ids = np.array([ord(c) % 256 for c in text][:8], dtype=np.int64)
if ids.shape[0] < 8:
    ids = np.pad(ids, (0, 8 - ids.shape[0]), constant_values=0)
ids = ids[None, :]
out = mlmodel.predict({'ids': ids})
print('token_id', int(ids[0,0]))
"""
    )
    console.print(f"Saved Core ML model to {mlmodel_path}")
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    dev = detect_device(args.device)
    model_mem = estimate_model_mem_gb(args.model, dev.backend)
    console.print(device_banner(__version__, dev, model_name=args.model, model_mem_gb=model_mem))
    framework = args.framework.lower()
    if framework == "coreml":
        try:
            return _convert_coreml(args)
        except Exception as e:
            console.print(f"[red]Convert failed:[/red] {e}")
            return 1
    console.print(f"Unsupported framework: {framework}")
    return 1


def cmd_optimize_llm(args: argparse.Namespace) -> int:
    dev = detect_device(args.device)
    model_mem = estimate_model_mem_gb(args.model, dev.backend)
    console.print(device_banner(__version__, dev, model_name=args.model, model_mem_gb=model_mem))
    # Load a small model and enable flash attention if available.
    # On MLX or MPS, it should log SDPA fallback.
    try:
        from transformers import AutoModelForCausalLM
    except Exception as e:
        console.print("Transformers is required.")
        return 1
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model)
    except Exception:
        model = None
    opt = LLMOptimizer(model)
    which = opt.enable_flash_attention(dev)
    console.print(f"Flash attention backend: {which}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="opt", description="OpenPostTrainingOptimizations CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    # status
    sp = sub.add_parser("status", help="Show device status")
    sp.add_argument("--device", default="auto")
    sp.add_argument("--model", default=None)
    sp.add_argument("-e", "--export", default="outputs/status.json")
    sp.set_defaults(func=cmd_status)

    # quantize
    sp = sub.add_parser("quantize", help="Quantize a model")
    sp.add_argument("-m", "--model", required=True)
    sp.add_argument("--method", default="int8", help="int8|int4|q8|q4|gptq|awq")
    sp.add_argument("-o", "--output", required=True)
    sp.add_argument("--device", default="auto")
    sp.set_defaults(func=cmd_quantize)

    # sparsify
    sp = sub.add_parser("sparsify", help="Apply sparsity/pruning to a model")
    sp.add_argument("-m", "--model", required=True)
    sp.add_argument("-s", "--sparsity", type=float, default=0.5)
    sp.add_argument("--pattern", default="2:4")
    sp.add_argument("-o", "--output", required=True)
    sp.add_argument("--device", default="auto")
    sp.set_defaults(func=cmd_sparsify)

    # profile
    sp = sub.add_parser("profile", help="Profile model performance")
    sp.add_argument("-m", "--model", required=True)
    sp.add_argument("--profile", nargs="+", default=["latency", "throughput", "memory"], choices=["latency", "throughput", "memory"])
    sp.add_argument("-b", "--batch-size", type=int, default=1)
    sp.add_argument("-i", "--iters", type=int, default=5)
    sp.add_argument("--input-len", type=int, default=32)
    sp.add_argument("-e", "--export", default="outputs/profile.json")
    sp.add_argument("--device", default="auto")
    sp.set_defaults(func=cmd_profile)

    # speculative
    sp = sub.add_parser("speculative", help="Speculative decoding benchmark")
    sp.add_argument("-m", "--model", required=True)
    sp.add_argument("-d", "--draft", required=True)
    sp.add_argument("--gamma", type=int, default=4)
    sp.add_argument("--benchmark", action="store_true")
    sp.add_argument("-o", "--output", default="outputs/speculative")
    sp.add_argument("--device", default="auto")
    sp.set_defaults(func=cmd_speculative)

    # serve
    sp = sub.add_parser("serve", help="Serve a model")
    sp.add_argument("-m", "--model", required=True, help="Model ID or GGUF path")
    sp.add_argument("--backend", choices=["mlx", "llama_cpp", "vllm"], default=None)
    sp.add_argument("--port", type=int, default=8000)
    sp.add_argument("--device", default="auto")
    sp.set_defaults(func=cmd_serve)

    # convert
    sp = sub.add_parser("convert", help="Convert a model to a different framework")
    sp.add_argument("-m", "--model", required=True)
    sp.add_argument("--framework", choices=["coreml"], required=True)
    sp.add_argument("-o", "--output", required=True)
    sp.add_argument("--device", default="auto")
    sp.set_defaults(func=cmd_convert)

    # optimize-llm
    sp = sub.add_parser("optimize-llm", help="Enable kernel-level optimizations (Flash-Attn/SDPA)")
    sp.add_argument("-m", "--model", required=True)
    sp.add_argument("-o", "--options", default="all", help="Optimization bundle to enable (default: all)")
    sp.add_argument("--device", default="auto")
    sp.set_defaults(func=cmd_optimize_llm)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
