from openposttraining.deployment.inference_engine import InferenceEngine


def test_streaming_yields_tokens():
    eng = InferenceEngine("sshleifer/tiny-gpt2", prefer_device="cpu")
    eng.load_model()
    chunks = []
    for i, ch in enumerate(eng.generate("Hello", max_tokens=4, stream=True)):
        chunks.append(ch)
        if i > 10:
            break
    assert len(chunks) >= 1
