import pytest

from openposttraining.agents.tools import AgentTools


def test_agent_tools_get_model_info():
    info = AgentTools.get_model_info("gpt2")
    assert info["name"] == "gpt2"
    assert info["status"] == "available"
    assert "backend" in info
    assert "size_gb" in info


def test_agent_tools_quantize_model():
    result = AgentTools.quantize_model("gpt2", method="int8", output_path="/tmp/test")
    assert result["model"] == "gpt2"
    assert result["method"] == "int8"
    assert result["output"] == "/tmp/test"
    assert result["status"] == "success"


def test_agent_tools_quantize_model_defaults():
    result = AgentTools.quantize_model("gpt2")
    assert result["model"] == "gpt2"
    assert result["method"] == "int8"
    assert "outputs/" in result["output"]


def test_agent_tools_profile_model():
    result = AgentTools.profile_model("gpt2")
    assert result["model"] == "gpt2"
    assert "latency_ms" in result
    assert "throughput_tps" in result
    assert "memory_gb" in result


def test_agent_tools_profile_model_custom_metrics():
    result = AgentTools.profile_model("gpt2", metrics=["latency", "memory"])
    assert "metrics" in result
    assert "latency" in result["metrics"]


def test_agent_tools_list_models():
    models = AgentTools.list_models()
    assert isinstance(models, list)
    assert len(models) > 0
    assert "gpt2" in models


def test_agent_tools_get_tool_definitions():
    tools = AgentTools.get_tool_definitions()
    assert isinstance(tools, list)
    assert len(tools) > 0
    assert all("type" in tool for tool in tools)
    assert all(tool["type"] == "function" for tool in tools)


def test_agent_tools_get_tool_map():
    tool_map = AgentTools.get_tool_map()
    assert isinstance(tool_map, dict)
    assert "get_model_info" in tool_map
    assert "quantize_model" in tool_map
    assert "profile_model" in tool_map
    assert "list_models" in tool_map
    assert callable(tool_map["get_model_info"])


def test_agent_tools_definitions_have_required_fields():
    tools = AgentTools.get_tool_definitions()
    for tool in tools:
        assert "function" in tool
        assert "name" in tool["function"]
        assert "description" in tool["function"]
        assert "parameters" in tool["function"]

