import pytest

from openposttraining.utils.hardware_utils import detect_device


@pytest.mark.cuda
def test_cuda_marker_skips_without_cuda():
    dev = detect_device("cuda")
    if dev.backend != "cuda":
        pytest.skip("CUDA not available on this runner")
    assert dev.backend == "cuda"
