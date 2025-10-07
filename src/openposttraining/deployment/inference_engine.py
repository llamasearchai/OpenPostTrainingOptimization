from __future__ import annotations

import threading
from typing import Generator, Optional

from ..utils.hardware_utils import DeviceInfo, detect_device, dev_synchronize, to_device

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


class InferenceEngine:
    def __init__(self, model_name: str, prefer_device: Optional[str] = None) -> None:
        self.model_name = model_name
        self.device: DeviceInfo = detect_device(prefer_device)
        self.model = None
        self.tokenizer = None
        self._backend = self.device.backend

    def load_model(self) -> None:
        if self._backend == "mlx":
            from mlx_lm import load

            self.model, self.tokenizer = load(self.model_name)
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model = to_device(self.model, self.device)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 32,
        stream: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ):
        if self.model is None or self.tokenizer is None:
            self.load_model()

        if stream:
            return self._generate_stream(prompt, max_tokens, temperature, top_p)
        return self._generate_sync(prompt, max_tokens, temperature, top_p)

    def _generate_sync(
        self, prompt: str, max_tokens: int, temperature: float, top_p: float
    ) -> str:
        """Generate text synchronously (non-streaming)."""
        if self._backend == "mlx":
            from mlx_lm import generate

            return generate(
                self.model,
                self.tokenizer,
                prompt,
                max_tokens=max_tokens,
                temp=temperature,
                top_p=top_p,
            )

        # Transformers path
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch is not None:
            inputs = {k: v.to(self.device.device_str) for k, v in inputs.items()}

        # Determine if we should sample
        do_sample = temperature != 1.0 or top_p != 1.0
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        with torch.no_grad():  # type: ignore[union-attr]
            out = self.model.generate(**inputs, **gen_kwargs)
        dev_synchronize(self._backend)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    def _generate_stream(
        self, prompt: str, max_tokens: int, temperature: float, top_p: float
    ):
        """Generate text in streaming mode (generator)."""
        if self._backend == "mlx":
            from mlx_lm import generate

            # Streaming: yield in small chunks
            text = generate(
                self.model,
                self.tokenizer,
                prompt,
                max_tokens=max_tokens,
                temp=temperature,
                top_p=top_p,
            )
            step = max(1, len(text) // 8)
            for i in range(0, len(text), step):
                yield text[i : i + step]
            return

        # Transformers path
        from transformers import TextIteratorStreamer

        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch is not None:
            inputs = {k: v.to(self.device.device_str) for k, v in inputs.items()}

        # Determine if we should sample
        do_sample = temperature != 1.0 or top_p != 1.0
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs["streamer"] = streamer
        thread = threading.Thread(target=self.model.generate, kwargs={**inputs, **gen_kwargs})
        thread.start()
        for new_text in streamer:  # type: ignore[attr-defined]
            yield new_text
        thread.join()
        dev_synchronize(self._backend)

    def unload(self) -> None:
        """Unload the model and tokenizer from memory."""
        self.model = None
        self.tokenizer = None


__all__ = ["InferenceEngine"]
