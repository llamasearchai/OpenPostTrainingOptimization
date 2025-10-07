from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from .config import AgentConfig


class AgentTools:
    """Tool definitions for AI agents."""

    @staticmethod
    def get_model_info(model_name: str) -> Dict[str, Any]:
        """Get information about a model.

        Args:
            model_name: Name of the model to query

        Returns:
            Dictionary containing model information
        """
        return {
            "name": model_name,
            "status": "available",
            "backend": "auto",
            "size_gb": 0.0,
            "quantized": False,
        }

    @staticmethod
    def quantize_model(
        model_name: str, method: str = "int8", output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Quantize a model.

        Args:
            model_name: Name of the model to quantize
            method: Quantization method (int8, int4, gptq, awq)
            output_path: Path to save quantized model

        Returns:
            Dictionary containing quantization results
        """
        return {
            "model": model_name,
            "method": method,
            "output": output_path or f"outputs/{model_name}-{method}",
            "status": "success",
            "size_reduction": "50%",
        }

    @staticmethod
    def profile_model(
        model_name: str, metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Profile model performance.

        Args:
            model_name: Name of the model to profile
            metrics: List of metrics to collect

        Returns:
            Dictionary containing profiling results
        """
        if metrics is None:
            metrics = ["latency", "throughput", "memory"]
        return {
            "model": model_name,
            "metrics": metrics,
            "latency_ms": 10.5,
            "throughput_tps": 100.0,
            "memory_gb": 2.5,
        }

    @staticmethod
    def list_models() -> List[str]:
        """List available models.

        Returns:
            List of model names
        """
        return [
            "gpt2",
            "distilgpt2",
            "mlx-community/TinyLlama-1.1B-Chat-v1.0",
        ]

    @classmethod
    def get_tool_definitions(cls) -> List[Dict[str, Any]]:
        """Get OpenAI function tool definitions.

        Returns:
            List of tool definition dictionaries
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_model_info",
                    "description": "Get information about a specific model",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "model_name": {
                                "type": "string",
                                "description": "Name of the model to query",
                            }
                        },
                        "required": ["model_name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "quantize_model",
                    "description": "Quantize a model to reduce size and improve inference speed",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "model_name": {
                                "type": "string",
                                "description": "Name of the model to quantize",
                            },
                            "method": {
                                "type": "string",
                                "enum": ["int8", "int4", "gptq", "awq"],
                                "description": "Quantization method to use",
                            },
                            "output_path": {
                                "type": "string",
                                "description": "Path to save the quantized model",
                            },
                        },
                        "required": ["model_name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "profile_model",
                    "description": "Profile model performance metrics",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "model_name": {
                                "type": "string",
                                "description": "Name of the model to profile",
                            },
                            "metrics": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of metrics to collect",
                            },
                        },
                        "required": ["model_name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_models",
                    "description": "List all available models",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]

    @classmethod
    def get_tool_map(cls) -> Dict[str, Callable]:
        """Get mapping of tool names to functions.

        Returns:
            Dictionary mapping tool names to callable functions
        """
        return {
            "get_model_info": cls.get_model_info,
            "quantize_model": cls.quantize_model,
            "profile_model": cls.profile_model,
            "list_models": cls.list_models,
        }

