from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


class LLMToolkitIntegration:
    """Integration with Simon Willison's llm toolkit."""

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize llm toolkit integration.

        Args:
            db_path: Path to llm database
        """
        self.db_path = db_path or Path.home() / ".local" / "share" / "llm" / "logs.db"

    def _run_llm_command(self, args: List[str]) -> str:
        """Run llm CLI command.

        Args:
            args: Command arguments

        Returns:
            Command output
        """
        try:
            result = subprocess.run(
                ["llm"] + args,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            return f"Error: {e.stderr}"
        except FileNotFoundError:
            return "Error: llm CLI not found. Install with: pip install llm"

    def list_models(self) -> str:
        """List available llm models.

        Returns:
            List of models as text
        """
        return self._run_llm_command(["models"])

    def prompt(
        self,
        prompt_text: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Send a prompt to llm.

        Args:
            prompt_text: Prompt text
            model: Model to use (e.g., "ollama:llama3")
            system: System prompt
            temperature: Temperature for generation

        Returns:
            Model response
        """
        args = []
        if model:
            args.extend(["-m", model])
        if system:
            args.extend(["-s", system])
        if temperature is not None:
            args.extend(["-o", f"temperature={temperature}"])
        args.append(prompt_text)
        return self._run_llm_command(args)

    def chat(
        self,
        model: Optional[str] = None,
        system: Optional[str] = None,
    ) -> str:
        """Start an llm chat session (non-interactive for API use).

        Args:
            model: Model to use
            system: System prompt

        Returns:
            Chat session info
        """
        args = ["chat"]
        if model:
            args.extend(["-m", model])
        if system:
            args.extend(["-s", system])
        return f"Chat session would start with: llm {' '.join(args)}"

    def embeddings(self, text: str, model: Optional[str] = None) -> str:
        """Generate embeddings using llm.

        Args:
            text: Text to embed
            model: Embedding model to use

        Returns:
            Embedding output
        """
        args = ["embed"]
        if model:
            args.extend(["-m", model])
        args.append(text)
        return self._run_llm_command(args)

    def list_logs(self, limit: int = 10) -> str:
        """List recent llm logs.

        Args:
            limit: Number of logs to retrieve

        Returns:
            Log entries
        """
        return self._run_llm_command(["logs", "list", "-n", str(limit)])

    def install_plugin(self, plugin_name: str) -> str:
        """Install an llm plugin.

        Args:
            plugin_name: Name of plugin to install

        Returns:
            Installation status
        """
        return self._run_llm_command(["install", plugin_name])

    def list_plugins(self) -> str:
        """List installed llm plugins.

        Returns:
            List of plugins
        """
        return self._run_llm_command(["plugins"])

    def run_cmd(self, command: str, context: Optional[str] = None) -> str:
        """Run llm-cmd to generate shell commands.

        Args:
            command: Natural language command description
            context: Additional context

        Returns:
            Generated command
        """
        try:
            args = ["llm-cmd", command]
            if context:
                args.extend(["--context", context])
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            return f"Error: {e.stderr}"
        except FileNotFoundError:
            return "Error: llm-cmd not found. Install with: pip install llm-cmd"

    def setup_ollama_plugin(self) -> str:
        """Setup llm-ollama plugin.

        Returns:
            Setup status
        """
        return self.install_plugin("llm-ollama")

