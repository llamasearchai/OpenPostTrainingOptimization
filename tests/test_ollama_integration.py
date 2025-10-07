import pytest
from unittest.mock import Mock, MagicMock, patch

from openposttraining.integrations import OllamaIntegration


def test_ollama_integration_init():
    ollama = OllamaIntegration()
    assert ollama.base_url == "http://localhost:11434"


def test_ollama_integration_custom_url():
    ollama = OllamaIntegration(base_url="http://custom:8080")
    assert ollama.base_url == "http://custom:8080"


@patch('openposttraining.integrations.ollama_integration.ollama')
def test_ollama_list_models_no_client(mock_ollama):
    mock_client = Mock()
    mock_client.list.return_value = {"models": [{"name": "llama3"}]}
    mock_ollama.Client.return_value = mock_client

    ollama = OllamaIntegration()
    models = ollama.list_models()
    assert isinstance(models, list)
    assert len(models) == 1


@patch('openposttraining.integrations.ollama_integration.ollama')
def test_ollama_pull_model_no_client(mock_ollama):
    mock_client = Mock()
    mock_client.pull.return_value = None  # pull returns None on success
    mock_ollama.Client.return_value = mock_client

    ollama = OllamaIntegration()
    result = ollama.pull_model("llama3")
    assert isinstance(result, dict)
    assert result["status"] == "success"


@patch('openposttraining.integrations.ollama_integration.ollama')
def test_ollama_delete_model_no_client(mock_ollama):
    mock_client = Mock()
    mock_client.delete.return_value = None
    mock_ollama.Client.return_value = mock_client

    ollama = OllamaIntegration()
    result = ollama.delete_model("test-model")
    assert isinstance(result, dict)
    assert result["status"] == "success"


@patch('openposttraining.integrations.ollama_integration.ollama')
def test_ollama_generate_no_client(mock_ollama):
    mock_client = Mock()
    mock_client.generate.return_value = {"response": "Generated text"}
    mock_ollama.Client.return_value = mock_client

    ollama = OllamaIntegration()
    result = ollama.generate("llama3", "Hello")
    assert isinstance(result, str)
    assert result == "Generated text"


@patch('openposttraining.integrations.ollama_integration.ollama')
def test_ollama_chat_no_client(mock_ollama):
    mock_client = Mock()
    mock_client.chat.return_value = {"message": {"content": "Chat response"}}
    mock_ollama.Client.return_value = mock_client

    ollama = OllamaIntegration()
    messages = [{"role": "user", "content": "Hello"}]
    result = ollama.chat("llama3", messages)
    assert isinstance(result, str)
    assert result == "Chat response"


@patch('openposttraining.integrations.ollama_integration.ollama')
def test_ollama_embeddings_no_client(mock_ollama):
    mock_client = Mock()
    mock_client.embeddings.return_value = {"embedding": [0.1, 0.2, 0.3]}
    mock_ollama.Client.return_value = mock_client

    ollama = OllamaIntegration()
    result = ollama.embeddings("llama3", "test text")
    assert isinstance(result, list)
    assert len(result) == 3


@patch('openposttraining.integrations.ollama_integration.ollama')
def test_ollama_show_model_info_no_client(mock_ollama):
    mock_client = Mock()
    mock_client.show.return_value = {"name": "llama3", "size": "7B"}
    mock_ollama.Client.return_value = mock_client

    ollama = OllamaIntegration()
    result = ollama.show_model_info("llama3")
    assert isinstance(result, dict)
    assert "name" in result


@patch('openposttraining.integrations.ollama_integration.ollama')
def test_ollama_generate_with_parameters(mock_ollama):
    mock_client = Mock()
    mock_client.generate.return_value = {"response": "Generated with params"}
    mock_ollama.Client.return_value = mock_client

    ollama = OllamaIntegration()
    result = ollama.generate(
        "llama3",
        "Hello",
        system="You are helpful",
        temperature=0.5,
        stream=False,
    )
    assert isinstance(result, str)
    assert result == "Generated with params"


@patch('openposttraining.integrations.ollama_integration.ollama')
def test_ollama_chat_with_temperature(mock_ollama):
    mock_client = Mock()
    mock_client.chat.return_value = {"message": {"content": "Chat with temp"}}
    mock_ollama.Client.return_value = mock_client

    ollama = OllamaIntegration()
    messages = [{"role": "user", "content": "Hello"}]
    result = ollama.chat("llama3", messages, temperature=0.8, stream=False)
    assert isinstance(result, str)
    assert result == "Chat with temp"

