import pytest

from openposttraining.models.llm_optimizer import LLMOptimizer
from openposttraining.utils.hardware_utils import DeviceInfo


def test_llm_optimizer_init_none():
    optimizer = LLMOptimizer(None)
    assert optimizer.model is None


def test_llm_optimizer_enable_flash_attention_cuda():
    device = DeviceInfo(
        backend="cuda",
        device_str="cuda:0",
        supports_flash_attn=True,
        supports_bnb=True,
        rss_gb=2.0,
        total_ram_gb=16.0,
    )
    optimizer = LLMOptimizer(None)
    result = optimizer.enable_flash_attention(device)
    assert result == "cuda"


def test_llm_optimizer_enable_flash_attention_mps():
    device = DeviceInfo(
        backend="mps",
        device_str="mps",
        supports_flash_attn=False,
        supports_bnb=False,
        rss_gb=2.0,
        total_ram_gb=16.0,
    )
    optimizer = LLMOptimizer(None)
    result = optimizer.enable_flash_attention(device)
    assert result == "sdpa"


def test_llm_optimizer_enable_flash_attention_cpu():
    device = DeviceInfo(
        backend="cpu",
        device_str="cpu",
        supports_flash_attn=False,
        supports_bnb=False,
        rss_gb=1.0,
        total_ram_gb=8.0,
    )
    optimizer = LLMOptimizer(None)
    result = optimizer.enable_flash_attention(device)
    assert result == "sdpa"


def test_llm_optimizer_compile_no_model():
    device = DeviceInfo(
        backend="cuda",
        device_str="cuda:0",
        supports_flash_attn=True,
        supports_bnb=True,
        rss_gb=2.0,
        total_ram_gb=16.0,
    )
    optimizer = LLMOptimizer(None)
    result = optimizer.compile(device)
    assert result is False


def test_llm_optimizer_compile_mps():
    device = DeviceInfo(
        backend="mps",
        device_str="mps",
        supports_flash_attn=False,
        supports_bnb=False,
        rss_gb=2.0,
        total_ram_gb=16.0,
    )
    optimizer = LLMOptimizer(None)
    result = optimizer.compile(device)
    assert result is False


def test_llm_optimizer_enable_flash_attention_mlx():
    device = DeviceInfo(
        backend="mlx",
        device_str="mlx",
        supports_flash_attn=False,
        supports_bnb=False,
        rss_gb=2.0,
        total_ram_gb=16.0,
    )
    optimizer = LLMOptimizer(None)
    result = optimizer.enable_flash_attention(device)
    assert result == "sdpa"

