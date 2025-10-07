from __future__ import annotations

__all__ = [
    "TUIConfig",
    "TUIState",
]

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TUIConfig:
    """TUI configuration."""

    theme: str = "dark"
    animations_enabled: bool = True
    autosave: bool = True
    config_path: Path = Path.home() / ".config" / "openposttraining" / "tui.json"


@dataclass
class TUIState:
    """TUI runtime state."""

    current_model: Optional[str] = None
    current_device: str = "auto"
    last_command: Optional[str] = None
    command_history: list[str] = None

    def __post_init__(self):
        if self.command_history is None:
            self.command_history = []

