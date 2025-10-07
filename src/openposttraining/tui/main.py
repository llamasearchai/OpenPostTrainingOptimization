from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Label, Static

from openposttraining import __version__
from openposttraining.utils.hardware_utils import detect_device

from . import TUIConfig, TUIState


class WelcomeScreen(Screen):
    """Welcome screen with menu options."""

    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True),
        Binding("1", "quantize", "Quantize Model"),
        Binding("2", "sparsify", "Apply Sparsity"),
        Binding("3", "profile", "Profile Performance"),
        Binding("4", "serve", "Serve Model"),
        Binding("5", "agents", "AI Agents"),
        Binding("6", "datasette", "Datasette Browser"),
        Binding("s", "settings", "Settings"),
    ]

    def __init__(self, config: TUIConfig, state: TUIState):
        super().__init__()
        self.config = config
        self.state = state

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Header()
        yield Container(
            Static(f"OpenPostTrainingOptimizations v{__version__}", id="title"),
            Static("Portable post-training optimization toolkit", id="subtitle"),
            Vertical(
                Label("Main Menu", classes="menu-title"),
                Button("[1] Quantize Model", id="quantize", variant="primary"),
                Button("[2] Apply Sparsity", id="sparsify", variant="primary"),
                Button("[3] Profile Performance", id="profile", variant="primary"),
                Button("[4] Serve Model", id="serve", variant="primary"),
                Button("[5] AI Agents & Chat", id="agents", variant="success"),
                Button("[6] Datasette Browser", id="datasette", variant="success"),
                Button("[S] Settings", id="settings"),
                Button("[Q] Quit", id="quit", variant="error"),
                id="menu",
            ),
            Horizontal(
                Static(f"Device: {self.state.current_device}", id="device-status"),
                Static(f"Model: {self.state.current_model or 'None'}", id="model-status"),
                id="status-bar",
            ),
            id="welcome-container",
        )
        yield Footer()

    @on(Button.Pressed, "#quantize")
    def action_quantize(self) -> None:
        """Navigate to quantization screen."""
        self.app.push_screen("quantize")

    @on(Button.Pressed, "#sparsify")
    def action_sparsify(self) -> None:
        """Navigate to sparsity screen."""
        self.app.push_screen("sparsify")

    @on(Button.Pressed, "#profile")
    def action_profile(self) -> None:
        """Navigate to profiling screen."""
        self.app.push_screen("profile")

    @on(Button.Pressed, "#serve")
    def action_serve(self) -> None:
        """Navigate to serving screen."""
        self.app.push_screen("serve")

    @on(Button.Pressed, "#agents")
    def action_agents(self) -> None:
        """Navigate to AI agents screen."""
        self.app.push_screen("agents")

    @on(Button.Pressed, "#datasette")
    def action_datasette(self) -> None:
        """Navigate to Datasette browser screen."""
        self.app.push_screen("datasette")

    @on(Button.Pressed, "#settings")
    def action_settings(self) -> None:
        """Navigate to settings screen."""
        self.app.push_screen("settings")

    @on(Button.Pressed, "#quit")
    def action_quit(self) -> None:
        """Quit the application."""
        self.app.exit()


class QuantizeScreen(Screen):
    """Screen for model quantization."""

    BINDINGS = [
        Binding("escape", "app.pop_screen", "Back"),
        Binding("q", "app.pop_screen", "Back"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Static("Quantize Model", id="screen-title"),
            Label("Select quantization method:"),
            Button("INT8 Quantization", id="int8"),
            Button("INT4 Quantization", id="int4"),
            Button("GPTQ Quantization", id="gptq"),
            Button("AWQ Quantization", id="awq"),
            Button("Back", id="back"),
        )
        yield Footer()

    @on(Button.Pressed, "#back")
    def go_back(self) -> None:
        self.app.pop_screen()


class SparsifyScreen(Screen):
    """Screen for applying sparsity."""

    BINDINGS = [
        Binding("escape", "app.pop_screen", "Back"),
        Binding("q", "app.pop_screen", "Back"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Static("Apply Sparsity", id="screen-title"),
            Label("Select sparsity pattern:"),
            Button("Unstructured 50%", id="unstructured"),
            Button("2:4 Structured", id="2-4"),
            Button("4:8 Structured", id="4-8"),
            Button("Back", id="back"),
        )
        yield Footer()

    @on(Button.Pressed, "#back")
    def go_back(self) -> None:
        self.app.pop_screen()


class ProfileScreen(Screen):
    """Screen for performance profiling."""

    BINDINGS = [
        Binding("escape", "app.pop_screen", "Back"),
        Binding("q", "app.pop_screen", "Back"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Static("Profile Performance", id="screen-title"),
            Label("Select profiling metrics:"),
            Button("Latency Profiling", id="latency"),
            Button("Throughput Profiling", id="throughput"),
            Button("Memory Profiling", id="memory"),
            Button("Full Profile", id="full"),
            Button("Back", id="back"),
        )
        yield Footer()

    @on(Button.Pressed, "#back")
    def go_back(self) -> None:
        self.app.pop_screen()


class ServeScreen(Screen):
    """Screen for model serving."""

    BINDINGS = [
        Binding("escape", "app.pop_screen", "Back"),
        Binding("q", "app.pop_screen", "Back"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Static("Serve Model", id="screen-title"),
            Label("Select serving backend:"),
            Button("MLX Server (Apple Silicon)", id="mlx"),
            Button("llama.cpp Server", id="llamacpp"),
            Button("vLLM Server (CUDA)", id="vllm"),
            Button("Back", id="back"),
        )
        yield Footer()

    @on(Button.Pressed, "#back")
    def go_back(self) -> None:
        self.app.pop_screen()


class AgentsScreen(Screen):
    """Screen for AI agents integration."""

    BINDINGS = [
        Binding("escape", "app.pop_screen", "Back"),
        Binding("q", "app.pop_screen", "Back"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Static("AI Agents & Chat", id="screen-title"),
            Label("AI Integration Options:"),
            Button("OpenAI Agents SDK", id="openai-agents"),
            Button("Ollama Chat", id="ollama-chat"),
            Button("LLM Toolkit", id="llm-toolkit"),
            Button("Agent Workflows", id="workflows"),
            Button("Back", id="back"),
        )
        yield Footer()

    @on(Button.Pressed, "#back")
    def go_back(self) -> None:
        self.app.pop_screen()


class DatasetteScreen(Screen):
    """Screen for Datasette integration."""

    BINDINGS = [
        Binding("escape", "app.pop_screen", "Back"),
        Binding("q", "app.pop_screen", "Back"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Static("Datasette Browser", id="screen-title"),
            Label("Data Management:"),
            Button("Browse Datasets", id="browse"),
            Button("Query Data", id="query"),
            Button("View Metrics", id="metrics"),
            Button("Export Data", id="export"),
            Button("Back", id="back"),
        )
        yield Footer()

    @on(Button.Pressed, "#back")
    def go_back(self) -> None:
        self.app.pop_screen()


class SettingsScreen(Screen):
    """Settings screen."""

    BINDINGS = [
        Binding("escape", "app.pop_screen", "Back"),
        Binding("q", "app.pop_screen", "Back"),
    ]

    def __init__(self, config: TUIConfig):
        super().__init__()
        self.config = config

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Static("Settings", id="screen-title"),
            Label(f"Theme: {self.config.theme}"),
            Label(f"Animations: {'Enabled' if self.config.animations_enabled else 'Disabled'}"),
            Label(f"Autosave: {'Enabled' if self.config.autosave else 'Disabled'}"),
            Button("Toggle Theme", id="toggle-theme"),
            Button("Toggle Animations", id="toggle-animations"),
            Button("Toggle Autosave", id="toggle-autosave"),
            Button("Save Settings", id="save"),
            Button("Back", id="back"),
        )
        yield Footer()

    @on(Button.Pressed, "#toggle-theme")
    def toggle_theme(self) -> None:
        self.config.theme = "light" if self.config.theme == "dark" else "dark"
        self.refresh()

    @on(Button.Pressed, "#toggle-animations")
    def toggle_animations(self) -> None:
        self.config.animations_enabled = not self.config.animations_enabled
        self.refresh()

    @on(Button.Pressed, "#toggle-autosave")
    def toggle_autosave(self) -> None:
        self.config.autosave = not self.config.autosave
        self.refresh()

    @on(Button.Pressed, "#save")
    def save_settings(self) -> None:
        self.config.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.config_path, "w") as f:
            json.dump(
                {
                    "theme": self.config.theme,
                    "animations_enabled": self.config.animations_enabled,
                    "autosave": self.config.autosave,
                },
                f,
                indent=2,
            )
        self.notify("Settings saved successfully")

    @on(Button.Pressed, "#back")
    def go_back(self) -> None:
        self.app.pop_screen()


class OpenPostTrainingTUI(App):
    """Main TUI application."""

    CSS = """
    #welcome-container {
        align: center middle;
        width: 80;
        height: auto;
    }

    #title {
        text-align: center;
        color: $accent;
        text-style: bold;
        margin: 1;
    }

    #subtitle {
        text-align: center;
        color: $text-muted;
        margin-bottom: 2;
    }

    #menu {
        align: center middle;
        width: 60;
        padding: 1;
    }

    .menu-title {
        text-align: center;
        text-style: bold;
        margin: 1;
    }

    Button {
        width: 100%;
        margin: 1;
    }

    #status-bar {
        dock: bottom;
        height: 3;
        background: $panel;
        padding: 1;
    }

    #device-status, #model-status {
        width: 50%;
        padding: 0 2;
    }

    #screen-title {
        text-align: center;
        text-style: bold;
        color: $accent;
        margin: 2;
    }

    Container {
        align: center top;
        padding: 2;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("ctrl+q", "quit", "Quit", priority=True),
    ]

    def __init__(self):
        super().__init__()
        self.config = self.load_config()
        self.state = TUIState()
        device = detect_device("auto")
        self.state.current_device = device.backend

    def load_config(self) -> TUIConfig:
        """Load configuration from disk."""
        config = TUIConfig()
        if config.config_path.exists():
            try:
                with open(config.config_path) as f:
                    data = json.load(f)
                config.theme = data.get("theme", "dark")
                config.animations_enabled = data.get("animations_enabled", True)
                config.autosave = data.get("autosave", True)
            except Exception:
                pass
        return config

    def on_mount(self) -> None:
        """Set up the application."""
        self.title = "OpenPostTrainingOptimizations"
        self.sub_title = f"v{__version__}"
        self.install_screen(WelcomeScreen(self.config, self.state), name="welcome")
        self.install_screen(QuantizeScreen(), name="quantize")
        self.install_screen(SparsifyScreen(), name="sparsify")
        self.install_screen(ProfileScreen(), name="profile")
        self.install_screen(ServeScreen(), name="serve")
        self.install_screen(AgentsScreen(), name="agents")
        self.install_screen(DatasetteScreen(), name="datasette")
        self.install_screen(SettingsScreen(self.config), name="settings")
        self.push_screen("welcome")


def app() -> None:
    """Run the TUI application."""
    tui = OpenPostTrainingTUI()
    tui.run()


if __name__ == "__main__":
    app()

