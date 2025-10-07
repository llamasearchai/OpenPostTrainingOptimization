import pytest

from openposttraining.deployment.inference_engine import InferenceEngine


def test_inference_engine_init():
    engine = InferenceEngine("sshleifer/tiny-gpt2", prefer_device="cpu")
    assert engine.model_name == "sshleifer/tiny-gpt2"
    assert engine.device.backend in ["cpu", "mps", "mlx", "cuda"]


def test_inference_engine_load_model():
    engine = InferenceEngine("sshleifer/tiny-gpt2", prefer_device="cpu")
    engine.load_model()
    assert engine.model is not None
    assert engine.tokenizer is not None


def test_inference_engine_generate_simple():
    engine = InferenceEngine("sshleifer/tiny-gpt2", prefer_device="cpu")
    engine.load_model()
    result = engine.generate("Hello", max_tokens=5)
    assert isinstance(result, str)
    assert len(result) > 0


def test_inference_engine_generate_streaming():
    engine = InferenceEngine("sshleifer/tiny-gpt2", prefer_device="cpu")
    engine.load_model()
    chunks = list(engine.generate("Hello", max_tokens=3, stream=True))
    assert len(chunks) > 0
    assert all(isinstance(chunk, str) for chunk in chunks)


def test_inference_engine_unload():
    engine = InferenceEngine("sshleifer/tiny-gpt2", prefer_device="cpu")
    engine.load_model()
    engine.unload()
    assert engine.model is None
    assert engine.tokenizer is None


def test_inference_engine_generate_temperature():
    engine = InferenceEngine("sshleifer/tiny-gpt2", prefer_device="cpu")
    engine.load_model()
    result = engine.generate("Hello", max_tokens=5, temperature=0.9)
    assert isinstance(result, str)


def test_inference_engine_generate_top_p():
    engine = InferenceEngine("sshleifer/tiny-gpt2", prefer_device="cpu")
    engine.load_model()
    result = engine.generate("Hello", max_tokens=5, top_p=0.9)
    assert isinstance(result, str)

