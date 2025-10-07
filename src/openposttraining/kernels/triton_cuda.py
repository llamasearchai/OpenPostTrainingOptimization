"""CUDA/Triton-only kernels.

This module must only be imported when running on CUDA.
Imports of `triton` are lazy and guarded to avoid errors on non-CUDA hosts.
"""
from __future__ import annotations

from typing import Any


def _import_triton() -> Any:
    try:
        import triton  # type: ignore

        return triton
    except Exception as e:  # pragma: no cover - only on CUDA
        raise RuntimeError("Triton not available. This path requires CUDA.") from e


def triton_available() -> bool:
    try:
        _import_triton()
        return True
    except Exception:
        return False


class FlashAttentionCUDA:
    """Placeholder wrapper.

    In real CUDA environments, implement Triton/flash-attn kernels here.
    """

    def __init__(self) -> None:  # pragma: no cover - exercised on CUDA only
        if not triton_available():
            raise RuntimeError("FlashAttentionCUDA requires Triton on CUDA.")

    def __repr__(self) -> str:  # pragma: no cover
        return "FlashAttentionCUDA(triton)"
