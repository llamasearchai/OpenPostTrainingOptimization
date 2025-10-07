import os
import pytest

from openposttraining.utils.hardware_utils import detect_device
from openposttraining.core.quantization import quantize_model


@pytest.mark.cuda
def test_cuda_int8_quantization(tmp_path):
    dev = detect_device("cuda")
    if dev.backend != "cuda":
        pytest.skip("CUDA not available")
    out = tmp_path / "q"
    res = quantize_model("sshleifer/tiny-gpt2", method="int8", output_dir=str(out), device=dev)
    assert out.exists()
    assert res.quantized_size_mb is None or res.original_size_mb is None or res.quantized_size_mb < res.original_size_mb
