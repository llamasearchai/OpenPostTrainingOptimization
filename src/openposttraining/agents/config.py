from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class AgentConfig:
    """Configuration for AI agents."""

    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o"
    openai_base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    streaming: bool = True
    tools_enabled: bool = True
    vector_store_enabled: bool = False
    file_search_enabled: bool = False
    retrieval_enabled: bool = False
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    storage_path: Path = Path.cwd() / "data" / "agents"

    def __post_init__(self):
        if self.openai_api_key is None:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.storage_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> AgentConfig:
        """Create config from environment variables."""
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            openai_base_url=os.getenv("OPENAI_BASE_URL"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "llama3"),
        )

