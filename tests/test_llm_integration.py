import pytest

from openposttraining.integrations import LLMToolkitIntegration


def test_llm_toolkit_init():
    llm = LLMToolkitIntegration()
    assert llm.db_path is not None


def test_llm_toolkit_custom_db_path(tmp_path):
    db_path = tmp_path / "test.db"
    llm = LLMToolkitIntegration(db_path=db_path)
    assert llm.db_path == db_path


def test_llm_toolkit_list_models():
    llm = LLMToolkitIntegration()
    result = llm.list_models()
    assert isinstance(result, str)


def test_llm_toolkit_prompt():
    llm = LLMToolkitIntegration()
    result = llm.prompt("Hello", model="gpt-3.5-turbo")
    assert isinstance(result, str)


def test_llm_toolkit_prompt_with_system():
    llm = LLMToolkitIntegration()
    result = llm.prompt("Hello", system="You are helpful", temperature=0.7)
    assert isinstance(result, str)


def test_llm_toolkit_chat():
    llm = LLMToolkitIntegration()
    result = llm.chat(model="gpt-3.5-turbo")
    assert isinstance(result, str)
    assert "llm chat" in result


def test_llm_toolkit_embeddings():
    llm = LLMToolkitIntegration()
    result = llm.embeddings("test text")
    assert isinstance(result, str)


def test_llm_toolkit_embeddings_with_model():
    llm = LLMToolkitIntegration()
    result = llm.embeddings("test text", model="text-embedding-ada-002")
    assert isinstance(result, str)


def test_llm_toolkit_list_logs():
    llm = LLMToolkitIntegration()
    result = llm.list_logs(limit=5)
    assert isinstance(result, str)


def test_llm_toolkit_list_plugins():
    llm = LLMToolkitIntegration()
    result = llm.list_plugins()
    assert isinstance(result, str)


def test_llm_toolkit_install_plugin():
    llm = LLMToolkitIntegration()
    result = llm.install_plugin("llm-ollama")
    assert isinstance(result, str)


def test_llm_toolkit_run_cmd():
    llm = LLMToolkitIntegration()
    result = llm.run_cmd("list files")
    assert isinstance(result, str)


def test_llm_toolkit_run_cmd_with_context():
    llm = LLMToolkitIntegration()
    result = llm.run_cmd("list files", context="in current directory")
    assert isinstance(result, str)


def test_llm_toolkit_setup_ollama_plugin():
    llm = LLMToolkitIntegration()
    result = llm.setup_ollama_plugin()
    assert isinstance(result, str)

