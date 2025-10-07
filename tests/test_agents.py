import pytest

from openposttraining.agents import AgentConfig, AgentRunner, create_agent


def test_agent_config_defaults():
    config = AgentConfig()
    assert config.openai_model == "gpt-4o"
    assert config.temperature == 0.7
    assert config.max_tokens == 4096
    assert config.streaming is True
    assert config.tools_enabled is True
    assert config.ollama_model == "llama3"


def test_agent_config_from_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-3.5-turbo")
    monkeypatch.setenv("OLLAMA_MODEL", "llama2")

    config = AgentConfig.from_env()
    assert config.openai_api_key == "test-key"
    assert config.openai_model == "gpt-3.5-turbo"
    assert config.ollama_model == "llama2"


def test_agent_runner_init():
    config = AgentConfig()
    runner = AgentRunner(config)
    assert runner.config == config
    assert runner.conversation_history == []


def test_agent_runner_reset():
    runner = AgentRunner()
    runner.conversation_history = [{"role": "user", "content": "test"}]
    runner.reset()
    assert runner.conversation_history == []


def test_create_agent():
    agent = create_agent()
    assert isinstance(agent, AgentRunner)
    assert agent.config is not None


def test_agent_chat_no_api_key():
    config = AgentConfig(openai_api_key=None)
    runner = AgentRunner(config)
    response = runner.chat("Hello")
    assert "Error" in response or "OPENAI_API_KEY" in response


def test_agent_runner_with_custom_config():
    config = AgentConfig(
        openai_model="custom-model",
        temperature=0.5,
        max_tokens=2048,
        streaming=False,
        tools_enabled=False,
    )
    runner = AgentRunner(config)
    assert runner.config.openai_model == "custom-model"
    assert runner.config.temperature == 0.5
    assert runner.config.max_tokens == 2048
    assert runner.config.streaming is False
    assert runner.config.tools_enabled is False

