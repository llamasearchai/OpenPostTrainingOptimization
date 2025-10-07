import pytest

from openposttraining.core.sparsity import sparsify_entry
from openposttraining.utils.hardware_utils import detect_device


def test_sparsity_2_4(tmp_path):
    dev = detect_device("cpu")  # device-agnostic
    out = tmp_path / "sparsity"
    # Use a model with wider dimensions that support 2:4 pattern (needs dim % 4 == 0)
    metrics = sparsify_entry("hf-internal-testing/tiny-random-gpt2", sparsity=0.5, pattern="2:4", output_dir=str(out), device=dev)
    assert out.exists()
    assert 0.3 <= metrics["actual_sparsity"] <= 0.7
    assert metrics["speedup_theoretical"] >= 1.5
