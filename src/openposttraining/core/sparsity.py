from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401 - used by potential advanced patterns

from ..utils.hardware_utils import DeviceInfo, ensure_dir, to_device


def _sparsify_linear_weight(weight: torch.Tensor, sparsity: float, pattern: str) -> Tuple[int, int]:
    total = weight.numel()
    pruned = 0
    if pattern == "2:4":
        # Group by 4 along the last dim and zero the smallest 2
        orig_shape = weight.shape
        last = orig_shape[-1]
        groups = last // 4 * 4
        if groups <= 0:
            return 0, 0
        w = weight[..., :groups].contiguous().view(-1, 4)
        # compute magnitude and keep top2 per group
        idx = torch.argsort(w.abs(), dim=1)
        # zero the smallest 2
        mask = torch.ones_like(w, dtype=torch.bool)
        mask.scatter_(1, idx[:, :2], False)
        pruned = int((~mask).sum().item())
        w = w * mask
        weight[..., :groups] = w.view(*orig_shape[:-1], groups)
        return pruned, total
    else:
        # Unstructured: zero out smallest fraction globally
        k = int(total * sparsity)
        if k <= 0:
            return 0, total
        flat = weight.view(-1)
        thresh = torch.topk(flat.abs(), k, largest=False).values.max()
        mask = flat.abs() > thresh
        pruned = int((~mask).sum().item())
        flat[~mask] = 0
        return pruned, total


def apply_sparsity(
    model: nn.Module,
    sparsity: float = 0.5,
    pattern: str = "2:4",
    dataloader: Optional[Any] = None,
) -> Dict[str, float]:
    total = 0
    pruned = 0
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Linear):
                p, t = _sparsify_linear_weight(m.weight, sparsity, pattern)
                total += t
                pruned += p
    actual = pruned / max(total, 1)
    # Optional fine-tuning step
    if dataloader is not None:
        # Keep optional and light; can be extended by user
        pass
    # Theoretical speedup for 2:4 is ~1.8-2.0x; approximate generally
    speedup = 2.0 if pattern == "2:4" else (1.0 / (1.0 - sparsity))
    return {"target_sparsity": sparsity, "actual_sparsity": actual, "speedup_theoretical": speedup}


def sparsify_entry(
    model_name: str,
    sparsity: float,
    pattern: str,
    output_dir: str,
    device: DeviceInfo,
) -> Dict[str, float]:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = to_device(model, device)
    metrics = apply_sparsity(model, sparsity=sparsity, pattern=pattern)
    ensure_dir(output_dir)
    with open(Path(output_dir) / "sparsity_config.json", "w") as f:
        json.dump(metrics, f)
    return metrics
