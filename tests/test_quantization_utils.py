import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from openposttraining.core.quantization import (
    QuantizationResult,
    _param_count,
    _dtype_bytes_default,
    _method_to_bits,
    _print_results,
    _save_config,
    _dir_size_mb,
)


def test_quantization_result_dataclass():
    result = QuantizationResult(
        backend="cpu",
        method="int8",
        original_size_mb=100.0,
        quantized_size_mb=50.0,
        speed_toks_per_s=10.5,
        speedup=2.0,
        output_dir="/tmp/test",
    )
    assert result.backend == "cpu"
    assert result.method == "int8"
    assert result.original_size_mb == 100.0
    assert result.quantized_size_mb == 50.0
    assert result.speed_toks_per_s == 10.5
    assert result.speedup == 2.0
    assert result.output_dir == "/tmp/test"


def test_quantization_result_with_none_values():
    result = QuantizationResult(
        backend="cpu",
        method="int8",
        original_size_mb=None,
        quantized_size_mb=None,
        speed_toks_per_s=None,
        speedup=None,
        output_dir="/tmp/test",
    )
    assert result.backend == "cpu"
    assert result.original_size_mb is None
    assert result.quantized_size_mb is None


def test_param_count_torch_model():
    """Test _param_count with torch model."""
    with patch('openposttraining.core.quantization.torch') as mock_torch:
        mock_model = Mock()
        mock_model.parameters.return_value = [
            Mock(numel=lambda: 100),
            Mock(numel=lambda: 200),
        ]
        mock_torch.nn.Module = type('Module', (), {})

        result = _param_count(mock_model)
        assert result == 300


def test_param_count_no_torch():
    """Test _param_count when torch is not available."""
    mock_model = Mock()
    result = _param_count(mock_model)
    # Should return None or handle gracefully
    assert result is None or isinstance(result, int)


def test_dtype_bytes_default():
    """Test _dtype_bytes_default function."""
    assert _dtype_bytes_default("mps") == 2
    assert _dtype_bytes_default("mlx") == 2
    assert _dtype_bytes_default("cuda") == 2
    assert _dtype_bytes_default("cpu") == 4


def test_method_to_bits():
    """Test _method_to_bits function."""
    assert _method_to_bits("int4") == 4
    assert _method_to_bits("int8") == 8
    assert _method_to_bits("q4") == 4
    assert _method_to_bits("q8") == 8
    assert _method_to_bits("gptq") == 8
    assert _method_to_bits("awq") == 8


@patch('openposttraining.core.quantization.console')
def test_print_results(mock_console):
    """Test _print_results function."""
    result = QuantizationResult(
        backend="cpu",
        method="int8",
        original_size_mb=100.0,
        quantized_size_mb=50.0,
        speed_toks_per_s=10.5,
        speedup=2.0,
        output_dir="/tmp/test",
    )
    _print_results(result, "test-model")
    mock_console.print.assert_called()


@patch('openposttraining.core.quantization.console')
def test_print_results_with_none_values(mock_console):
    """Test _print_results with None values."""
    result = QuantizationResult(
        backend="cpu",
        method="int8",
        original_size_mb=None,
        quantized_size_mb=None,
        speed_toks_per_s=None,
        speedup=None,
        output_dir="/tmp/test",
    )
    _print_results(result, "test-model")
    mock_console.print.assert_called()


def test_save_config(tmp_path):
    """Test _save_config function."""
    result = QuantizationResult(
        backend="cpu",
        method="int8",
        original_size_mb=100.0,
        quantized_size_mb=50.0,
        speed_toks_per_s=10.5,
        speedup=2.0,
        output_dir=str(tmp_path),
    )
    _save_config(result, str(tmp_path))

    config_file = tmp_path / "quantization_config.json"
    assert config_file.exists()
    with open(config_file) as f:
        data = json.load(f)
    assert data["method"] == "int8"
    assert data["backend"] == "cpu"


def test_dir_size_mb(tmp_path):
    """Test _dir_size_mb function."""
    # Create some test files
    (tmp_path / "file1.txt").write_text("x" * 1024)  # 1KB
    (tmp_path / "file2.txt").write_text("y" * 2048)  # 2KB

    size = _dir_size_mb(tmp_path)
    assert size is not None
    assert size > 0


def test_dir_size_mb_empty_dir(tmp_path):
    """Test _dir_size_mb with empty directory."""
    size = _dir_size_mb(tmp_path)
    assert size == 0.0


def test_dir_size_mb_nonexistent():
    """Test _dir_size_mb with nonexistent directory."""
    size = _dir_size_mb(Path("/nonexistent/path"))
    assert size is None

