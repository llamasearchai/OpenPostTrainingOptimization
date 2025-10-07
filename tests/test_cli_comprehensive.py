import argparse
import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from openposttraining.cli import (
    build_parser,
    cmd_status,
    cmd_quantize,
    cmd_sparsify,
    cmd_profile,
    cmd_speculative,
    cmd_serve,
    cmd_convert,
    cmd_optimize_llm,
    _convert_coreml,
    main,
)


def test_cli_parser_creation():
    parser = build_parser()
    assert parser is not None


def test_cli_status_parsing():
    parser = build_parser()
    args = parser.parse_args(["status", "--device", "cpu"])
    assert args.cmd == "status"
    assert args.device == "cpu"


def test_cli_quantize_parsing():
    parser = build_parser()
    args = parser.parse_args([
        "quantize",
        "-m", "gpt2",
        "--method", "int8",
        "-o", "outputs/test",
        "--device", "cpu",
    ])
    assert args.cmd == "quantize"
    assert args.model == "gpt2"
    assert args.method == "int8"
    assert args.output == "outputs/test"
    assert args.device == "cpu"


def test_cli_sparsify_parsing():
    parser = build_parser()
    args = parser.parse_args([
        "sparsify",
        "-m", "gpt2",
        "-s", "0.5",
        "--pattern", "2:4",
        "-o", "outputs/test",
    ])
    assert args.cmd == "sparsify"
    assert args.model == "gpt2"
    assert args.sparsity == 0.5
    assert args.pattern == "2:4"


def test_cli_profile_parsing():
    parser = build_parser()
    args = parser.parse_args([
        "profile",
        "-m", "gpt2",
        "--profile", "latency", "throughput",
    ])
    assert args.cmd == "profile"
    assert args.model == "gpt2"
    assert "latency" in args.profile
    assert "throughput" in args.profile


def test_cli_speculative_parsing():
    parser = build_parser()
    args = parser.parse_args([
        "speculative",
        "-m", "gpt2",
        "-d", "distilgpt2",
        "--gamma", "4",
        "--benchmark",
    ])
    assert args.cmd == "speculative"
    assert args.model == "gpt2"
    assert args.draft == "distilgpt2"
    assert args.gamma == 4
    assert args.benchmark is True


def test_cli_serve_parsing():
    parser = build_parser()
    args = parser.parse_args([
        "serve",
        "-m", "gpt2",
        "--backend", "llama_cpp",
        "--port", "8080",
    ])
    assert args.cmd == "serve"
    assert args.model == "gpt2"
    assert args.backend == "llama_cpp"
    assert args.port == 8080


def test_cli_convert_parsing():
    parser = build_parser()
    args = parser.parse_args([
        "convert",
        "-m", "gpt2",
        "--framework", "coreml",
        "-o", "outputs/test",
    ])
    assert args.cmd == "convert"
    assert args.model == "gpt2"
    assert args.framework == "coreml"


def test_cli_optimize_llm_parsing():
    parser = build_parser()
    args = parser.parse_args([
        "optimize-llm",
        "-m", "gpt2",
        "-o", "all",
    ])
    assert args.cmd == "optimize-llm"
    assert args.model == "gpt2"
    assert args.options == "all"


def test_cli_profile_default_metrics():
    parser = build_parser()
    args = parser.parse_args(["profile", "-m", "gpt2"])
    assert "latency" in args.profile
    assert "throughput" in args.profile
    assert "memory" in args.profile


def test_cli_status_with_export():
    parser = build_parser()
    args = parser.parse_args(["status", "-e", "custom/path.json"])
    assert args.export == "custom/path.json"


def test_cli_quantize_default_method():
    parser = build_parser()
    args = parser.parse_args(["quantize", "-m", "gpt2", "-o", "out"])
    assert args.method == "int8"


def test_cli_sparsify_default_pattern():
    parser = build_parser()
    args = parser.parse_args(["sparsify", "-m", "gpt2", "-o", "out"])
    assert args.pattern == "2:4"
    assert args.sparsity == 0.5


def test_cli_speculative_default_gamma():
    parser = build_parser()
    args = parser.parse_args(["speculative", "-m", "gpt2", "-d", "distilgpt2", "-o", "out"])
    assert args.gamma == 4
    assert args.benchmark is False


def test_cli_serve_default_port():
    parser = build_parser()
    args = parser.parse_args(["serve", "-m", "gpt2"])
    assert args.port == 8000


def test_cli_profile_batch_size():
    parser = build_parser()
    args = parser.parse_args(["profile", "-m", "gpt2", "-b", "16"])
    assert args.batch_size == 16


def test_cli_profile_iters():
    parser = build_parser()
    args = parser.parse_args(["profile", "-m", "gpt2", "-i", "10"])
    assert args.iters == 10


# Command execution tests

@patch('openposttraining.cli.detect_device')
@patch('openposttraining.cli.estimate_model_mem_gb')
@patch('openposttraining.cli.device_banner')
@patch('openposttraining.cli.console')
def test_cmd_status(mock_console, mock_banner, mock_mem, mock_detect, tmp_path):
    """Test cmd_status command."""
    mock_dev = Mock(backend="cpu", device_str="cpu", rss_gb=1.0, total_ram_gb=16.0,
                   supports_flash_attn=False, supports_bnb=False)
    mock_detect.return_value = mock_dev
    mock_mem.return_value = 0.5
    mock_banner.return_value = "Device Banner"

    args = argparse.Namespace(device="cpu", model="gpt2", export=str(tmp_path / "status.json"))
    result = cmd_status(args)

    assert result == 0
    assert (tmp_path / "status.json").exists()
    with open(tmp_path / "status.json") as f:
        data = json.load(f)
    assert data["backend"] == "cpu"


@patch('openposttraining.cli.detect_device')
@patch('openposttraining.cli.estimate_model_mem_gb')
@patch('openposttraining.cli.device_banner')
@patch('openposttraining.cli.quantize_model')
@patch('openposttraining.cli.console')
def test_cmd_quantize_success(mock_console, mock_quant, mock_banner, mock_mem, mock_detect, tmp_path):
    """Test cmd_quantize command success."""
    mock_dev = Mock(backend="cpu", device_str="cpu")
    mock_detect.return_value = mock_dev
    mock_mem.return_value = 0.5
    mock_quant.return_value = Mock()

    args = argparse.Namespace(device="cpu", model="gpt2", method="int8", output=str(tmp_path))
    result = cmd_quantize(args)

    assert result == 0
    mock_quant.assert_called_once()


@patch('openposttraining.cli.detect_device')
@patch('openposttraining.cli.estimate_model_mem_gb')
@patch('openposttraining.cli.device_banner')
@patch('openposttraining.cli.quantize_model')
@patch('openposttraining.cli.console')
def test_cmd_quantize_failure(mock_console, mock_quant, mock_banner, mock_mem, mock_detect):
    """Test cmd_quantize command failure."""
    mock_dev = Mock(backend="cpu")
    mock_detect.return_value = mock_dev
    mock_mem.return_value = 0.5
    mock_quant.side_effect = Exception("Quantization error")

    args = argparse.Namespace(device="cpu", model="gpt2", method="int8", output="/tmp/out")
    result = cmd_quantize(args)

    assert result == 1


@patch('openposttraining.cli.detect_device')
@patch('openposttraining.cli.estimate_model_mem_gb')
@patch('openposttraining.cli.device_banner')
@patch('openposttraining.cli.sparsify_entry')
@patch('openposttraining.cli.console')
def test_cmd_sparsify_success(mock_console, mock_sparsify, mock_banner, mock_mem, mock_detect, tmp_path):
    """Test cmd_sparsify command success."""
    mock_dev = Mock(backend="cpu")
    mock_detect.return_value = mock_dev
    mock_mem.return_value = 0.5
    mock_sparsify.return_value = {"actual_sparsity": 0.5, "speedup_theoretical": 2.0}

    args = argparse.Namespace(device="cpu", model="gpt2", sparsity=0.5, pattern="2:4", output=str(tmp_path))
    result = cmd_sparsify(args)

    assert result == 0
    mock_sparsify.assert_called_once()


@patch('openposttraining.cli.detect_device')
@patch('openposttraining.cli.estimate_model_mem_gb')
@patch('openposttraining.cli.device_banner')
@patch('openposttraining.cli.sparsify_entry')
@patch('openposttraining.cli.console')
def test_cmd_sparsify_failure(mock_console, mock_sparsify, mock_banner, mock_mem, mock_detect):
    """Test cmd_sparsify command failure."""
    mock_dev = Mock(backend="cpu")
    mock_detect.return_value = mock_dev
    mock_mem.return_value = 0.5
    mock_sparsify.side_effect = Exception("Sparsify error")

    args = argparse.Namespace(device="cpu", model="gpt2", sparsity=0.5, pattern="2:4", output="/tmp/out")
    result = cmd_sparsify(args)

    assert result == 1


@patch('openposttraining.cli.detect_device')
@patch('openposttraining.cli.estimate_model_mem_gb')
@patch('openposttraining.cli.device_banner')
@patch('openposttraining.cli.profile_entry')
@patch('openposttraining.cli.console')
def test_cmd_profile_success(mock_console, mock_profile, mock_banner, mock_mem, mock_detect):
    """Test cmd_profile command success."""
    mock_dev = Mock(backend="cpu")
    mock_detect.return_value = mock_dev
    mock_mem.return_value = 0.5
    mock_profile.return_value = {
        "latency": {"mean": 10.5, "p95": 15.2},
        "throughput": {"tokens_per_second": 100.0, "samples_per_second": 50.0},
        "memory": {"peak_gb": 2.0, "model_gb": 1.0, "activation_gb": 0.5, "reserved_gb": 0.5}
    }

    args = argparse.Namespace(
        device="cpu", model="gpt2", profile=["latency", "throughput", "memory"],
        batch_size=1, iters=5, input_len=32, export="/tmp/profile.json"
    )
    result = cmd_profile(args)

    assert result == 0
    mock_profile.assert_called_once()


@patch('openposttraining.cli.detect_device')
@patch('openposttraining.cli.estimate_model_mem_gb')
@patch('openposttraining.cli.device_banner')
@patch('openposttraining.cli.profile_entry')
@patch('openposttraining.cli.console')
def test_cmd_profile_failure(mock_console, mock_profile, mock_banner, mock_mem, mock_detect):
    """Test cmd_profile command failure."""
    mock_dev = Mock(backend="cpu")
    mock_detect.return_value = mock_dev
    mock_mem.return_value = 0.5
    mock_profile.side_effect = Exception("Profile error")

    args = argparse.Namespace(
        device="cpu", model="gpt2", profile=["latency"],
        batch_size=1, iters=5, input_len=32, export="/tmp/profile.json"
    )
    result = cmd_profile(args)

    assert result == 1


@patch('openposttraining.cli.detect_device')
@patch('openposttraining.cli.estimate_model_mem_gb')
@patch('openposttraining.cli.device_banner')
@patch('openposttraining.cli.speculative_entry')
@patch('openposttraining.cli.console')
def test_cmd_speculative_success(mock_console, mock_spec, mock_banner, mock_mem, mock_detect, tmp_path):
    """Test cmd_speculative command success."""
    mock_dev = Mock(backend="cpu")
    mock_detect.return_value = mock_dev
    mock_mem.return_value = 0.5
    mock_spec.return_value = {"tokens_per_second": 50.0, "acceptance_rate": 0.8, "latency_ms": 100.0}

    args = argparse.Namespace(
        device="cpu", model="gpt2", draft="distilgpt2", gamma=4, benchmark=True, output=str(tmp_path)
    )
    result = cmd_speculative(args)

    assert result == 0
    mock_spec.assert_called_once()


@patch('openposttraining.cli.detect_device')
@patch('openposttraining.cli.estimate_model_mem_gb')
@patch('openposttraining.cli.device_banner')
@patch('openposttraining.cli.speculative_entry')
@patch('openposttraining.cli.console')
def test_cmd_speculative_failure(mock_console, mock_spec, mock_banner, mock_mem, mock_detect):
    """Test cmd_speculative command failure."""
    mock_dev = Mock(backend="cpu")
    mock_detect.return_value = mock_dev
    mock_mem.return_value = 0.5
    mock_spec.side_effect = Exception("Speculative error")

    args = argparse.Namespace(
        device="cpu", model="gpt2", draft="distilgpt2", gamma=4, benchmark=False, output="/tmp/out"
    )
    result = cmd_speculative(args)

    assert result == 1


@patch('openposttraining.cli.detect_device')
@patch('openposttraining.cli.estimate_model_mem_gb')
@patch('openposttraining.cli.device_banner')
@patch('openposttraining.cli.console')
def test_cmd_serve_vllm_warning(mock_console, mock_banner, mock_mem, mock_detect):
    """Test cmd_serve with vllm backend shows warning."""
    mock_dev = Mock(backend="mps")
    mock_detect.return_value = mock_dev
    mock_mem.return_value = 0.5

    args = argparse.Namespace(device="auto", model="gpt2", backend="vllm", port=8000)
    result = cmd_serve(args)

    assert result == 0


@patch('openposttraining.cli.detect_device')
@patch('openposttraining.cli.estimate_model_mem_gb')
@patch('openposttraining.cli.device_banner')
@patch('openposttraining.cli.console')
def test_cmd_serve_unknown_backend(mock_console, mock_banner, mock_mem, mock_detect):
    """Test cmd_serve with unknown backend."""
    mock_dev = Mock(backend="cpu")
    mock_detect.return_value = mock_dev
    mock_mem.return_value = 0.5

    args = argparse.Namespace(device="cpu", model="gpt2", backend="unknown", port=8000)
    result = cmd_serve(args)

    assert result == 1


@patch('openposttraining.cli.detect_device')
@patch('openposttraining.cli.estimate_model_mem_gb')
@patch('openposttraining.cli.device_banner')
@patch('openposttraining.cli._convert_coreml')
@patch('openposttraining.cli.console')
def test_cmd_convert_coreml_success(mock_console, mock_convert, mock_banner, mock_mem, mock_detect, tmp_path):
    """Test cmd_convert with CoreML."""
    mock_dev = Mock(backend="cpu")
    mock_detect.return_value = mock_dev
    mock_mem.return_value = 0.5
    mock_convert.return_value = 0

    args = argparse.Namespace(device="cpu", model="gpt2", framework="coreml", output=str(tmp_path))
    result = cmd_convert(args)

    assert result == 0


@patch('openposttraining.cli.detect_device')
@patch('openposttraining.cli.estimate_model_mem_gb')
@patch('openposttraining.cli.device_banner')
@patch('openposttraining.cli._convert_coreml')
@patch('openposttraining.cli.console')
def test_cmd_convert_coreml_failure(mock_console, mock_convert, mock_banner, mock_mem, mock_detect):
    """Test cmd_convert with CoreML failure."""
    mock_dev = Mock(backend="cpu")
    mock_detect.return_value = mock_dev
    mock_mem.return_value = 0.5
    mock_convert.side_effect = Exception("Convert error")

    args = argparse.Namespace(device="cpu", model="gpt2", framework="coreml", output="/tmp/out")
    result = cmd_convert(args)

    assert result == 1


@patch('openposttraining.cli.detect_device')
@patch('openposttraining.cli.estimate_model_mem_gb')
@patch('openposttraining.cli.device_banner')
@patch('openposttraining.cli.console')
def test_cmd_convert_unsupported_framework(mock_console, mock_banner, mock_mem, mock_detect):
    """Test cmd_convert with unsupported framework."""
    mock_dev = Mock(backend="cpu")
    mock_detect.return_value = mock_dev
    mock_mem.return_value = 0.5

    args = argparse.Namespace(device="cpu", model="gpt2", framework="onnx", output="/tmp/out")
    result = cmd_convert(args)

    assert result == 1


@patch('openposttraining.cli.detect_device')
@patch('openposttraining.cli.estimate_model_mem_gb')
@patch('openposttraining.cli.device_banner')
@patch('openposttraining.cli.LLMOptimizer')
@patch('openposttraining.cli.AutoModelForCausalLM')
@patch('openposttraining.cli.console')
def test_cmd_optimize_llm_success(mock_console, mock_model_class, mock_opt_class, mock_banner, mock_mem, mock_detect):
    """Test cmd_optimize_llm success."""
    mock_dev = Mock(backend="cpu")
    mock_detect.return_value = mock_dev
    mock_mem.return_value = 0.5
    mock_model = Mock()
    mock_model_class.from_pretrained.return_value = mock_model
    mock_opt = Mock()
    mock_opt.enable_flash_attention.return_value = "sdpa"
    mock_opt_class.return_value = mock_opt

    args = argparse.Namespace(device="cpu", model="gpt2", options="all")
    result = cmd_optimize_llm(args)

    assert result == 0


@patch('openposttraining.cli.detect_device')
@patch('openposttraining.cli.estimate_model_mem_gb')
@patch('openposttraining.cli.device_banner')
@patch('openposttraining.cli.AutoModelForCausalLM')
@patch('openposttraining.cli.console')
def test_cmd_optimize_llm_model_load_failure(mock_console, mock_model_class, mock_banner, mock_mem, mock_detect):
    """Test cmd_optimize_llm with model load failure."""
    mock_dev = Mock(backend="cpu")
    mock_detect.return_value = mock_dev
    mock_mem.return_value = 0.5
    mock_model_class.from_pretrained.side_effect = Exception("Model not found")

    args = argparse.Namespace(device="cpu", model="invalid-model", options="all")
    result = cmd_optimize_llm(args)

    # Should still return 0 because it handles None model
    assert result == 0


@patch('openposttraining.cli.torch')
@patch('openposttraining.cli.ensure_dir')
@patch('openposttraining.cli.console')
def test_convert_coreml_success(mock_console, mock_ensure, mock_torch, tmp_path):
    """Test _convert_coreml function success."""
    mock_ct = Mock()
    mock_torch.nn.Module = type('Module', (), {})
    mock_torch.nn.Embedding = type('Embedding', (mock_torch.nn.Module,), {'__init__': lambda s, v, d: None})
    mock_torch.jit.trace = Mock(return_value=Mock())

    with patch('openposttraining.cli.ct', mock_ct):
        mock_ct.convert.return_value = Mock(save=Mock())

        args = argparse.Namespace(output=str(tmp_path))
        result = _convert_coreml(args)

        assert result == 0


@patch('openposttraining.cli.torch')
@patch('openposttraining.cli.console')
def test_convert_coreml_missing_dependencies(mock_console, mock_torch):
    """Test _convert_coreml with missing dependencies."""
    mock_torch_import = Mock(side_effect=ImportError("No module named 'torch'"))

    with patch('builtins.__import__', mock_torch_import):
        args = argparse.Namespace(output="/tmp/out")
        result = _convert_coreml(args)

        assert result == 1


def test_main_with_args():
    """Test main function with arguments."""
    parser = build_parser()
    args = ["status", "--device", "cpu"]

    with patch('openposttraining.cli.cmd_status', return_value=0) as mock_cmd:
        result = main(args)
        assert result == 0


def test_main_default_argv():
    """Test main function with default argv."""
    with patch('sys.argv', ['opt', 'status']):
        with patch('openposttraining.cli.cmd_status', return_value=0):
            result = main()
            assert result == 0

