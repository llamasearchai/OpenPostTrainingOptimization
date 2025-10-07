import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from textual.widgets import Button

from openposttraining.tui import TUIConfig, TUIState
from openposttraining.tui.main import (
    WelcomeScreen,
    QuantizeScreen,
    SparsifyScreen,
    ProfileScreen,
    ServeScreen,
    AgentsScreen,
    DatasetteScreen,
    SettingsScreen,
    OpenPostTrainingTUI,
    app,
)


def test_tui_config_defaults():
    config = TUIConfig()
    assert config.theme == "dark"
    assert config.animations_enabled is True
    assert config.autosave is True
    assert "openposttraining" in str(config.config_path)


def test_tui_config_custom():
    config = TUIConfig(theme="light", animations_enabled=False)
    assert config.theme == "light"
    assert config.animations_enabled is False


def test_tui_state_defaults():
    state = TUIState()
    assert state.current_model is None
    assert state.current_device == "auto"
    assert state.last_command is None
    assert state.command_history == []


def test_tui_state_with_values():
    state = TUIState(
        current_model="gpt2",
        current_device="cuda",
        last_command="quantize",
        command_history=["status", "quantize"],
    )
    assert state.current_model == "gpt2"
    assert state.current_device == "cuda"
    assert state.last_command == "quantize"
    assert len(state.command_history) == 2


def test_tui_state_command_history_init():
    state = TUIState()
    state.command_history.append("test")
    assert "test" in state.command_history


def test_welcome_screen_init():
    config = TUIConfig()
    state = TUIState()
    screen = WelcomeScreen(config, state)
    assert screen.config == config
    assert screen.state == state


def test_welcome_screen_compose():
    config = TUIConfig()
    state = TUIState()
    screen = WelcomeScreen(config, state)
    widgets = list(screen.compose())
    assert len(widgets) > 0


def test_welcome_screen_action_quantize():
    config = TUIConfig()
    state = TUIState()
    screen = WelcomeScreen(config, state)
    screen.app = Mock()
    screen.action_quantize()
    screen.app.push_screen.assert_called_once_with("quantize")


def test_welcome_screen_action_sparsify():
    config = TUIConfig()
    state = TUIState()
    screen = WelcomeScreen(config, state)
    screen.app = Mock()
    screen.action_sparsify()
    screen.app.push_screen.assert_called_once_with("sparsify")


def test_welcome_screen_action_profile():
    config = TUIConfig()
    state = TUIState()
    screen = WelcomeScreen(config, state)
    screen.app = Mock()
    screen.action_profile()
    screen.app.push_screen.assert_called_once_with("profile")


def test_welcome_screen_action_serve():
    config = TUIConfig()
    state = TUIState()
    screen = WelcomeScreen(config, state)
    screen.app = Mock()
    screen.action_serve()
    screen.app.push_screen.assert_called_once_with("serve")


def test_welcome_screen_action_agents():
    config = TUIConfig()
    state = TUIState()
    screen = WelcomeScreen(config, state)
    screen.app = Mock()
    screen.action_agents()
    screen.app.push_screen.assert_called_once_with("agents")


def test_welcome_screen_action_datasette():
    config = TUIConfig()
    state = TUIState()
    screen = WelcomeScreen(config, state)
    screen.app = Mock()
    screen.action_datasette()
    screen.app.push_screen.assert_called_once_with("datasette")


def test_welcome_screen_action_settings():
    config = TUIConfig()
    state = TUIState()
    screen = WelcomeScreen(config, state)
    screen.app = Mock()
    screen.action_settings()
    screen.app.push_screen.assert_called_once_with("settings")


def test_welcome_screen_action_quit():
    config = TUIConfig()
    state = TUIState()
    screen = WelcomeScreen(config, state)
    screen.app = Mock()
    screen.action_quit()
    screen.app.exit.assert_called_once()


def test_quantize_screen_compose():
    screen = QuantizeScreen()
    widgets = list(screen.compose())
    assert len(widgets) > 0


def test_quantize_screen_go_back():
    screen = QuantizeScreen()
    screen.app = Mock()
    screen.go_back()
    screen.app.pop_screen.assert_called_once()


def test_sparsify_screen_compose():
    screen = SparsifyScreen()
    widgets = list(screen.compose())
    assert len(widgets) > 0


def test_sparsify_screen_go_back():
    screen = SparsifyScreen()
    screen.app = Mock()
    screen.go_back()
    screen.app.pop_screen.assert_called_once()


def test_profile_screen_compose():
    screen = ProfileScreen()
    widgets = list(screen.compose())
    assert len(widgets) > 0


def test_profile_screen_go_back():
    screen = ProfileScreen()
    screen.app = Mock()
    screen.go_back()
    screen.app.pop_screen.assert_called_once()


def test_serve_screen_compose():
    screen = ServeScreen()
    widgets = list(screen.compose())
    assert len(widgets) > 0


def test_serve_screen_go_back():
    screen = ServeScreen()
    screen.app = Mock()
    screen.go_back()
    screen.app.pop_screen.assert_called_once()


def test_agents_screen_compose():
    screen = AgentsScreen()
    widgets = list(screen.compose())
    assert len(widgets) > 0


def test_agents_screen_go_back():
    screen = AgentsScreen()
    screen.app = Mock()
    screen.go_back()
    screen.app.pop_screen.assert_called_once()


def test_datasette_screen_compose():
    screen = DatasetteScreen()
    widgets = list(screen.compose())
    assert len(widgets) > 0


def test_datasette_screen_go_back():
    screen = DatasetteScreen()
    screen.app = Mock()
    screen.go_back()
    screen.app.pop_screen.assert_called_once()


def test_settings_screen_init():
    config = TUIConfig()
    screen = SettingsScreen(config)
    assert screen.config == config


def test_settings_screen_compose():
    config = TUIConfig()
    screen = SettingsScreen(config)
    widgets = list(screen.compose())
    assert len(widgets) > 0


def test_settings_screen_toggle_theme():
    config = TUIConfig(theme="dark")
    screen = SettingsScreen(config)
    screen.refresh = Mock()
    screen.toggle_theme()
    assert config.theme == "light"
    screen.toggle_theme()
    assert config.theme == "dark"


def test_settings_screen_toggle_animations():
    config = TUIConfig(animations_enabled=True)
    screen = SettingsScreen(config)
    screen.refresh = Mock()
    screen.toggle_animations()
    assert config.animations_enabled is False
    screen.toggle_animations()
    assert config.animations_enabled is True


def test_settings_screen_toggle_autosave():
    config = TUIConfig(autosave=True)
    screen = SettingsScreen(config)
    screen.refresh = Mock()
    screen.toggle_autosave()
    assert config.autosave is False
    screen.toggle_autosave()
    assert config.autosave is True


def test_settings_screen_save_settings(tmp_path):
    config = TUIConfig(theme="light", animations_enabled=False, autosave=False)
    config.config_path = tmp_path / "config.json"
    screen = SettingsScreen(config)
    screen.notify = Mock()
    screen.save_settings()

    assert config.config_path.exists()
    with open(config.config_path) as f:
        data = json.load(f)
    assert data["theme"] == "light"
    assert data["animations_enabled"] is False
    assert data["autosave"] is False
    screen.notify.assert_called_once_with("Settings saved successfully")


def test_settings_screen_go_back():
    config = TUIConfig()
    screen = SettingsScreen(config)
    screen.app = Mock()
    screen.go_back()
    screen.app.pop_screen.assert_called_once()


@patch('openposttraining.tui.main.detect_device')
def test_openposttraining_tui_init(mock_detect):
    mock_detect.return_value = Mock(backend="cpu")
    tui = OpenPostTrainingTUI()
    assert tui.config is not None
    assert tui.state is not None
    assert tui.state.current_device == "cpu"


@patch('openposttraining.tui.main.detect_device')
def test_openposttraining_tui_load_config_no_file(mock_detect, tmp_path):
    mock_detect.return_value = Mock(backend="cpu")
    with patch.object(TUIConfig, 'config_path', tmp_path / "nonexistent.json"):
        tui = OpenPostTrainingTUI()
        assert tui.config.theme == "dark"
        assert tui.config.animations_enabled is True


@patch('openposttraining.tui.main.detect_device')
def test_openposttraining_tui_load_config_with_file(mock_detect, tmp_path):
    mock_detect.return_value = Mock(backend="cpu")
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({
        "theme": "light",
        "animations_enabled": False,
        "autosave": False
    }))

    with patch.object(TUIConfig, 'config_path', config_file):
        tui = OpenPostTrainingTUI()
        assert tui.config.theme == "light"
        assert tui.config.animations_enabled is False
        assert tui.config.autosave is False


@patch('openposttraining.tui.main.detect_device')
def test_openposttraining_tui_load_config_invalid_json(mock_detect, tmp_path):
    mock_detect.return_value = Mock(backend="cpu")
    config_file = tmp_path / "config.json"
    config_file.write_text("invalid json{")

    with patch.object(TUIConfig, 'config_path', config_file):
        tui = OpenPostTrainingTUI()
        # Should fall back to defaults
        assert tui.config.theme == "dark"


@patch('openposttraining.tui.main.detect_device')
def test_openposttraining_tui_on_mount(mock_detect):
    mock_detect.return_value = Mock(backend="cpu")
    tui = OpenPostTrainingTUI()
    tui.install_screen = Mock()
    tui.push_screen = Mock()

    tui.on_mount()

    assert tui.title == "OpenPostTrainingOptimizations"
    assert "v" in tui.sub_title
    assert tui.install_screen.call_count == 8
    tui.push_screen.assert_called_once_with("welcome")


@patch('openposttraining.tui.main.OpenPostTrainingTUI')
def test_app_function(mock_tui_class):
    mock_tui_instance = Mock()
    mock_tui_class.return_value = mock_tui_instance

    app()

    mock_tui_class.assert_called_once()
    mock_tui_instance.run.assert_called_once()

