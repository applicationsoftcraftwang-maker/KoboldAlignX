"""
Tests for configuration module.
"""

import pytest
from src.core.config import settings


def test_settings_load():
    """Test that settings load properly."""
    assert settings.app_name == "KoboldAlignX"
    assert settings.app_version is not None


def test_database_url_exists():
    """Test that database URL is configured."""
    # TODO: Implement with proper env file
    pass
