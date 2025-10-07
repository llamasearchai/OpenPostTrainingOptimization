from __future__ import annotations

import os
import platform
import shutil
from dataclasses import dataclass
from typing import Any, Optional

import psutil

try:
    import torch
except Exception:  # pragma: no cover - torch may not be installed in minimal env
    torch = None  # type: ignore


@dataclass
class DeviceInfo:
    backend: str  # "mlx" | "mps" | "cuda" | "cpu"
    device_str: str  # e.g. "mlx" | "mps" | "cuda:0" | "cpu"
    supports_flash_attn: bool
    supports_bnb: bool
    rss_gb: float
    total_ram_gb: float


def _rss_gb() -> float:
    try:
        proc = psutil.Process()
        return proc.memory_info().rss / 1e9
    except Exception:
        return 0.0


def _total_ram_gb() -> float:
    try:
        return psutil.virtual_memory().total / 1e9
    except Exception:
        return 0.0


def _has_mlx() -> bool:
    if platform.system() != "Darwin":
        return False
    try:
        import mlx.core  # noqa: F401

        return True
    except Exception:
        return False


def _has_mps() -> bool:
    if torch is None:
        return False
    try:
        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def _has_cuda() -> bool:
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _supports_flash_attn(backend: str) -> bool:
    if backend != "cuda":
        return False
    try:
        __import__("flash_attn")
        return True
    except Exception:
        pass
    try:
        __import__("triton")
        return True
    except Exception:
        return False


def _supports_bnb(backend: str) -> bool:
    if backend != "cuda":
        return False
    try:
        __import__("bitsandbytes")
        return True
    except Exception:
        return False


def detect_device(prefer: Optional[str] = None) -> DeviceInfo:
    """
    Detect the best backend to use.

    prefer can be one of: "auto"|None, "mlx", "mps", "cuda", "cpu".
    """
    env_pref = os.getenv("OPT_DEVICE")
    if env_pref:
        prefer = env_pref
    prefer = (prefer or "auto").lower()

    backend: str
    device_str: str

    if prefer == "mlx" and _has_mlx():
        backend, device_str = "mlx", "mlx"
    elif prefer == "mps" and _has_mps():
        backend, device_str = "mps", "mps"
    elif prefer == "cuda" and _has_cuda():
        backend, device_str = "cuda", "cuda:0"
    elif prefer == "cpu":
        backend, device_str = "cpu", "cpu"
    else:
        # auto selection
        if _has_mlx():
            backend, device_str = "mlx", "mlx"
        elif _has_mps():
            backend, device_str = "mps", "mps"
        elif _has_cuda():
            backend, device_str = "cuda", "cuda:0"
        else:
            backend, device_str = "cpu", "cpu"

    info = DeviceInfo(
        backend=backend,
        device_str=device_str,
        supports_flash_attn=_supports_flash_attn(backend),
        supports_bnb=_supports_bnb(backend),
        rss_gb=_rss_gb(),
        total_ram_gb=_total_ram_gb(),
    )
    return info


def dev_synchronize(backend: str) -> None:
    if backend == "cuda" and torch is not None:
        if hasattr(torch, "cuda") and hasattr(torch.cuda, "synchronize"):
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
    elif backend == "mps" and torch is not None:
        # torch.mps.synchronize may not exist on older versions
        try:
            if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
        except Exception:
            pass
    elif backend == "mlx":
        try:
            import mlx.core as mx

            mx.sync()
        except Exception:
            pass
    # CPU is no-op


def to_device(obj: Any, device: DeviceInfo) -> Any:
    """Move a torch tensor/model to the target device or return unchanged for MLX objects."""
    if device.backend == "mlx":
        # MLX uses its own array types; leave objects as-is.
        return obj
    if torch is None:
        return obj
    try:
        dev = device.device_str
        if isinstance(obj, torch.nn.Module):
            return obj.to(dev)
        if isinstance(obj, torch.Tensor):
            return obj.to(dev)
        return obj
    except Exception:
        return obj


def dev_tensor(obj: Any, device: DeviceInfo) -> Any:
    """Alias for to_device for compatibility with earlier code paths."""
    return to_device(obj, device)


def device_banner(
    version: str,
    device: DeviceInfo,
    model_name: Optional[str] = None,
    model_mem_gb: Optional[float] = None,
) -> str:
    ram = f"{device.rss_gb:.1f}/{device.total_ram_gb:.1f}GB" if device.total_ram_gb else "n/a"
    model = model_name or "-"
    extra = f" | ModelMem: {model_mem_gb:.2f} GB" if model_mem_gb is not None else ""
    line = f"│ Backend: {device.backend} | Model: {model} | RAM: {ram:<11}{extra} │"
    width = max(47, len(line) + 2)
    top = "┌" + "─" * (width - 2) + "┐"
    title = f"│ OpenPostTrainingOptimizations v{version:<9}".ljust(width - 1) + "│"
    body = line.ljust(width - 1) + "│"
    bottom = "└" + "─" * (width - 2) + "┘"
    return "\n".join([top, title, body, bottom])


def _approx_param_count_from_config(model_name: str) -> Optional[int]:
    try:
        from transformers import AutoConfig  # type: ignore

        cfg = AutoConfig.from_pretrained(model_name)
        hidden = int(getattr(cfg, "n_embd", getattr(cfg, "hidden_size", 0)))
        vocab = int(getattr(cfg, "vocab_size", 0))
        layers = int(getattr(cfg, "n_layer", getattr(cfg, "num_hidden_layers", 0)))
        if hidden and vocab and layers:
            return vocab * hidden + layers * (hidden * hidden * 12)
    except Exception:
        return None
    return None


def estimate_model_mem_gb(model_name: Optional[str], backend: str) -> Optional[float]:
    if not model_name:
        return None
    params = _approx_param_count_from_config(model_name)
    if not params:
        return None
    bytes_per = 2 if backend in {"mps", "mlx", "cuda"} else 4
    return params * bytes_per / (1024**3)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)
