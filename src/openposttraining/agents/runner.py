from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Optional

from .config import AgentConfig
from .tools import AgentTools


class AgentRunner:
    """Runner for AI agent interactions."""

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize agent runner.

        Args:
            config: Agent configuration (defaults to env-based config)
        """
        self.config = config or AgentConfig.from_env()
        self.tools = AgentTools()
        self.conversation_history: List[Dict[str, Any]] = []

    def _get_openai_client(self):
        """Get OpenAI client (lazy import)."""
        try:
            from openai import OpenAI

            return OpenAI(
                api_key=self.config.openai_api_key,
                base_url=self.config.openai_base_url,
            )
        except ImportError:
            raise ImportError(
                "OpenAI package is required. Install with: pip install openai"
            )

    def chat(self, message: str, stream: bool = False) -> Any:
        """Send a chat message to the agent.

        Args:
            message: User message
            stream: Whether to stream the response

        Returns:
            Agent response (string or stream)
        """
        if not self.config.openai_api_key:
            return "Error: OPENAI_API_KEY not set"

        client = self._get_openai_client()
        self.conversation_history.append({"role": "user", "content": message})

        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant for the OpenPostTrainingOptimizations toolkit. "
                "You help users optimize and serve machine learning models using quantization, sparsity, "
                "and other post-training optimization techniques.",
            }
        ] + self.conversation_history

        tools = self.tools.get_tool_definitions() if self.config.tools_enabled else None

        response = client.chat.completions.create(
            model=self.config.openai_model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            tools=tools,
            stream=stream,
        )

        if stream:
            return response

        # Handle tool calls
        if response.choices[0].message.tool_calls:
            return self._handle_tool_calls(response.choices[0].message, client, messages)

        assistant_message = response.choices[0].message.content
        self.conversation_history.append({"role": "assistant", "content": assistant_message})
        return assistant_message

    def _handle_tool_calls(self, message, client, messages: List[Dict[str, Any]]) -> str:
        """Handle function/tool calls from the agent.

        Args:
            message: Message with tool calls
            client: OpenAI client
            messages: Conversation messages

        Returns:
            Final response after tool execution
        """
        tool_map = self.tools.get_tool_map()
        messages.append(message.model_dump())

        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            if function_name in tool_map:
                result = tool_map[function_name](**function_args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result),
                    }
                )

        # Get final response
        final_response = client.chat.completions.create(
            model=self.config.openai_model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        assistant_message = final_response.choices[0].message.content
        self.conversation_history.append({"role": "assistant", "content": assistant_message})
        return assistant_message

    async def chat_async(self, message: str) -> str:
        """Async chat with the agent.

        Args:
            message: User message

        Returns:
            Agent response
        """
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                api_key=self.config.openai_api_key,
                base_url=self.config.openai_base_url,
            )
        except ImportError:
            raise ImportError(
                "OpenAI package is required. Install with: pip install openai"
            )

        self.conversation_history.append({"role": "user", "content": message})

        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant for the OpenPostTrainingOptimizations toolkit.",
            }
        ] + self.conversation_history

        response = await client.chat.completions.create(
            model=self.config.openai_model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        assistant_message = response.choices[0].message.content
        self.conversation_history.append({"role": "assistant", "content": assistant_message})
        return assistant_message

    async def stream_chat(self, message: str) -> AsyncIterator[str]:
        """Stream chat responses.

        Args:
            message: User message

        Yields:
            Chunks of the response
        """
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                api_key=self.config.openai_api_key,
                base_url=self.config.openai_base_url,
            )
        except ImportError:
            raise ImportError(
                "OpenAI package is required. Install with: pip install openai"
            )

        self.conversation_history.append({"role": "user", "content": message})

        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant for the OpenPostTrainingOptimizations toolkit.",
            }
        ] + self.conversation_history

        stream = await client.chat.completions.create(
            model=self.config.openai_model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=True,
        )

        full_response = ""
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                yield content

        self.conversation_history.append({"role": "assistant", "content": full_response})

    def reset(self) -> None:
        """Reset conversation history."""
        self.conversation_history = []


def create_agent(config: Optional[AgentConfig] = None) -> AgentRunner:
    """Create an agent runner instance.

    Args:
        config: Optional agent configuration

    Returns:
        AgentRunner instance
    """
    return AgentRunner(config)

