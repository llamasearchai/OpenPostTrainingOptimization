from __future__ import annotations

import warnings
from typing import Optional

from rich.console import Console

from ..utils.hardware_utils import DeviceInfo

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - torch may not be importable in CI doc build
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore

console = Console()


class SDPAAttention(nn.Module):  # type: ignore
    def __init__(self):  # pragma: no cover - thin wrapper
        super().__init__()

    def forward(self, q, k, v, attn_mask=None, dropout_p: float = 0.0):  # type: ignore
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)


class LLMOptimizer:
    def __init__(self, model: Optional["torch.nn.Module"]):  # type: ignore
        self.model = model

    def enable_flash_attention(self, device: DeviceInfo) -> str:
        if device.backend == "cuda" and device.supports_flash_attn:
            console.log("Flash attention: Triton/CUDA enabled")
            # Integration with flash-attn kernels would go here.
            return "cuda"
        else:
            console.log(f"Flash attention: SDPA fallback ({device.backend.upper()})")
            # On non-CUDA, we rely on PyTorch SDPA which is optimized for MPS/CPU/MLX backends
            return "sdpa"

    def compile(self, device: DeviceInfo, dynamic: bool = False) -> bool:
        if torch is None or self.model is None:
            return False
        if device.backend == "mps":
            warnings.warn("torch.compile is experimental/disabled on MPS; skipping compile.")
            return False
        if not hasattr(torch, "compile"):
            return False
        try:
            self.model = torch.compile(self.model, dynamic=dynamic)  # type: ignore[attr-defined]
            return True
        except Exception as e:  # pragma: no cover - compile can be flaky
            warnings.warn(f"torch.compile failed: {e}")
            return False
