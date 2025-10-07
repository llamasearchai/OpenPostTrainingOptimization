from __future__ import annotations

from typing import Optional

try:
    from fastapi import FastAPI
except Exception as e:  # pragma: no cover
    raise RuntimeError("FastAPI is required, install with [serve]") from e

from pydantic import BaseModel

from ...deployment.inference_engine import InferenceEngine


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 64


def create_app(model_name: str) -> FastAPI:
    engine = InferenceEngine(model_name, prefer_device="mlx")
    engine.load_model()
    app = FastAPI()

    @app.get("/healthz")
    def healthz():  # pragma: no cover - simple
        return {"status": "ok", "backend": "mlx"}

    @app.post("/generate")
    def generate(req: GenerateRequest):  # pragma: no cover - simple
        text = engine.generate(req.prompt, max_tokens=req.max_tokens or 64)
        return {"text": text}

    return app


__all__ = ["create_app"]
