import os
import pytest

from openposttraining.utils.hardware_utils import detect_device
from openposttraining.core.quantization import quantize_model

# Check if optimum-quanto is available
try:
    from optimum.quanto import QuantoConfig
    QUANTO_AVAILABLE = True
except ImportError:
    QUANTO_AVAILABLE = False


@pytest.mark.mps
@pytest.mark.skipif(not QUANTO_AVAILABLE, reason="optimum-quanto not installed")
def test_quanto_int8_quantization(tmp_path):
    dev = detect_device("mps")
    if dev.backend != "mps":
        pytest.skip("MPS not available")
    out = tmp_path / "q"
    res = quantize_model("sshleifer/tiny-gpt2", method="int8", output_dir=str(out), device=dev)
    assert out.exists()
    # size reduction vs assumed fp16 baseline
    assert res.quantized_size_mb is None or res.original_size_mb is None or res.quantized_size_mb < res.original_size_mb
