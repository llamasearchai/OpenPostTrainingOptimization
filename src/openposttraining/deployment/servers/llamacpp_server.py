from __future__ import annotations

from typing import Optional

try:
    from fastapi import FastAPI
except Exception as e:  # pragma: no cover
    raise RuntimeError("FastAPI is required, install with [serve]") from e

from pydantic import BaseModel


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 64


def create_app(model_path: str) -> FastAPI:
    try:
        from llama_cpp import Llama  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("llama-cpp-python is required for llama_cpp backend.") from e

    llm = Llama(model_path=model_path, n_ctx=2048, n_threads=0)
    app = FastAPI()

    @app.get("/healthz")
    def healthz():  # pragma: no cover
        return {"status": "ok", "backend": "llama_cpp"}

    @app.post("/generate")
    def generate(req: GenerateRequest):  # pragma: no cover
        out = llm(req.prompt, max_tokens=req.max_tokens or 64, echo=False)
        text = out.get("choices", [{}])[0].get("text", "")
        return {"text": text}

    return app


__all__ = ["create_app"]
