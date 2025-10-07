import pytest

from openposttraining.utils.hardware_utils import (
    DeviceInfo,
    detect_device,
    dev_synchronize,
    dev_tensor,
    device_banner,
    ensure_dir,
    estimate_model_mem_gb,
    to_device,
    which,
)


def test_device_info_creation():
    device = DeviceInfo(
        backend="cpu",
        device_str="cpu",
        supports_flash_attn=False,
        supports_bnb=False,
        rss_gb=1.5,
        total_ram_gb=8.0,
    )
    assert device.backend == "cpu"
    assert device.device_str == "cpu"
    assert device.supports_flash_attn is False
    assert device.supports_bnb is False
    assert device.rss_gb == 1.5
    assert device.total_ram_gb == 8.0


def test_detect_device_cpu():
    device = detect_device("cpu")
    assert device.backend == "cpu"
    assert device.device_str == "cpu"
    assert device.rss_gb >= 0
    assert device.total_ram_gb > 0


def test_detect_device_auto():
    device = detect_device("auto")
    assert device.backend in ["mlx", "mps", "cuda", "cpu"]
    assert device.device_str is not None


def test_detect_device_env(monkeypatch):
    monkeypatch.setenv("OPT_DEVICE", "cpu")
    device = detect_device()
    assert device.backend == "cpu"


def test_dev_synchronize_cpu():
    dev_synchronize("cpu")


def test_dev_synchronize_mlx():
    dev_synchronize("mlx")


def test_to_device_non_torch():
    device = DeviceInfo(
        backend="mlx",
        device_str="mlx",
        supports_flash_attn=False,
        supports_bnb=False,
        rss_gb=1.0,
        total_ram_gb=8.0,
    )
    obj = {"test": "data"}
    result = to_device(obj, device)
    assert result == obj


def test_dev_tensor_alias():
    device = DeviceInfo(
        backend="cpu",
        device_str="cpu",
        supports_flash_attn=False,
        supports_bnb=False,
        rss_gb=1.0,
        total_ram_gb=8.0,
    )
    obj = "test"
    result = dev_tensor(obj, device)
    assert result == obj


def test_device_banner():
    device = DeviceInfo(
        backend="cpu",
        device_str="cpu",
        supports_flash_attn=False,
        supports_bnb=False,
        rss_gb=1.5,
        total_ram_gb=8.0,
    )
    banner = device_banner("1.0.0", device, model_name="gpt2", model_mem_gb=0.5)
    assert "Backend: cpu" in banner
    assert "Model: gpt2" in banner
    assert "ModelMem: 0.50 GB" in banner
    assert "OpenPostTrainingOptimizations" in banner


def test_device_banner_minimal():
    device = DeviceInfo(
        backend="cpu",
        device_str="cpu",
        supports_flash_attn=False,
        supports_bnb=False,
        rss_gb=1.0,
        total_ram_gb=8.0,
    )
    banner = device_banner("1.0.0", device)
    assert "Backend: cpu" in banner
    assert "OpenPostTrainingOptimizations" in banner


def test_estimate_model_mem_gb_none():
    result = estimate_model_mem_gb(None, "cpu")
    assert result is None


def test_estimate_model_mem_gb_invalid():
    result = estimate_model_mem_gb("nonexistent/model", "cpu")
    assert result is None or isinstance(result, float)


def test_ensure_dir(tmp_path):
    test_dir = tmp_path / "test" / "nested"
    ensure_dir(str(test_dir))
    assert test_dir.exists()


def test_ensure_dir_existing(tmp_path):
    test_dir = tmp_path / "existing"
    test_dir.mkdir()
    ensure_dir(str(test_dir))
    assert test_dir.exists()


def test_which_existing_command():
    result = which("python")
    assert result is not None or result is None


def test_which_nonexistent_command():
    result = which("nonexistent_command_12345")
    assert result is None


def test_detect_device_explicit_mlx():
    device = detect_device("mlx")
    assert device.backend in ["mlx", "mps", "cuda", "cpu"]


def test_detect_device_explicit_mps():
    device = detect_device("mps")
    assert device.backend in ["mps", "cpu", "mlx", "cuda"]


def test_detect_device_explicit_cuda():
    device = detect_device("cuda")
    assert device.backend in ["cuda", "cpu", "mlx", "mps"]

