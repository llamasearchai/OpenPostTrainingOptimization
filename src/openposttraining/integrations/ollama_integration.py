from __future__ import annotations

from typing import Any, Dict, List, Optional


class OllamaIntegration:
    """Integration with Ollama for local model serving."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        """Initialize Ollama integration.

        Args:
            base_url: Base URL for Ollama API
        """
        self.base_url = base_url
        self._client = None

    def _get_client(self):
        """Get Ollama client (lazy import)."""
        if self._client is None:
            try:
                import ollama

                self._client = ollama.Client(host=self.base_url)
            except ImportError:
                raise ImportError(
                    "Ollama package is required. Install with: pip install ollama"
                )
        return self._client

    def list_models(self) -> List[Dict[str, Any]]:
        """List available Ollama models.

        Returns:
            List of model information dictionaries
        """
        try:
            client = self._get_client()
            response = client.list()
            return response.get("models", [])
        except Exception as e:
            return [{"error": str(e)}]

    def pull_model(self, model_name: str) -> Dict[str, Any]:
        """Pull a model from Ollama registry.

        Args:
            model_name: Name of the model to pull

        Returns:
            Status information
        """
        try:
            client = self._get_client()
            client.pull(model_name)
            return {"status": "success", "model": model_name}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def delete_model(self, model_name: str) -> Dict[str, Any]:
        """Delete a model from local Ollama.

        Args:
            model_name: Name of the model to delete

        Returns:
            Status information
        """
        try:
            client = self._get_client()
            client.delete(model_name)
            return {"status": "success", "model": model_name}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def generate(
        self,
        model_name: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> Any:
        """Generate text using Ollama model.

        Args:
            model_name: Name of the model to use
            prompt: User prompt
            system: System prompt
            temperature: Temperature for generation
            stream: Whether to stream the response

        Returns:
            Generated text or stream
        """
        try:
            client = self._get_client()
            response = client.generate(
                model=model_name,
                prompt=prompt,
                system=system,
                options={"temperature": temperature},
                stream=stream,
            )
            if stream:
                return response
            return response.get("response", "")
        except Exception as e:
            return f"Error: {e}"

    def chat(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        stream: bool = False,
    ) -> Any:
        """Chat with Ollama model.

        Args:
            model_name: Name of the model to use
            messages: List of message dictionaries
            temperature: Temperature for generation
            stream: Whether to stream the response

        Returns:
            Chat response or stream
        """
        try:
            client = self._get_client()
            response = client.chat(
                model=model_name,
                messages=messages,
                options={"temperature": temperature},
                stream=stream,
            )
            if stream:
                return response
            return response.get("message", {}).get("content", "")
        except Exception as e:
            return f"Error: {e}"

    def embeddings(self, model_name: str, text: str) -> List[float]:
        """Generate embeddings using Ollama.

        Args:
            model_name: Name of the embedding model
            text: Text to embed

        Returns:
            List of embedding values
        """
        try:
            client = self._get_client()
            response = client.embeddings(model=model_name, prompt=text)
            return response.get("embedding", [])
        except Exception as e:
            return []

    def show_model_info(self, model_name: str) -> Dict[str, Any]:
        """Show information about a model.

        Args:
            model_name: Name of the model

        Returns:
            Model information dictionary
        """
        try:
            client = self._get_client()
            response = client.show(model_name)
            return response
        except Exception as e:
            return {"error": str(e)}

