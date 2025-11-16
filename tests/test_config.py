"""Tests for configuration module."""
import pytest

from src.core.config import Settings, get_settings, reload_settings


class TestSettings:
    """Test settings configuration."""

    def test_settings_singleton(self):
        """Test that settings is a singleton."""
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2

    def test_settings_reload(self):
        """Test settings reload."""
        settings1 = get_settings()
        settings2 = reload_settings()
        assert settings1 is not settings2

    def test_default_settings(self):
        """Test default settings values."""
        settings = get_settings()
        assert settings.app_name == "ALPR System"
        assert settings.app_version == "2.0.0"
        assert settings.environment in ["development", "staging", "production"]

    def test_settings_directories(self):
        """Test that directory paths are set."""
        settings = get_settings()
        assert settings.data_dir is not None
        assert settings.config_dir is not None
        assert settings.logs_dir is not None

    def test_ensure_directories(self):
        """Test that ensure_directories creates required directories."""
        settings = get_settings()
        settings.ensure_directories()
        # If this doesn't raise an exception, the test passes
