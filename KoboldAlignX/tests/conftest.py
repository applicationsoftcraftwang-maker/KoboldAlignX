"""
Pytest configuration and shared fixtures.

Provides common fixtures for testing all services.
"""

import os
from typing import AsyncGenerator, Dict, List

import aiohttp
import pandas as pd
import pytest
from unittest.mock import AsyncMock, MagicMock, Mock

# Set test environment
os.environ['ENVIRONMENT'] = 'test'


@pytest.fixture
def sample_job() -> Dict:
    """Sample job data from Reliance API."""
    return {
        "jobId": 28238,
        "location": "AB_100-01-02-003-04W5",
        "status": "completed",
        "name": "Test Job"
    }


@pytest.fixture
def sample_jobs() -> List[Dict]:
    """Sample list of jobs."""
    return [
        {"jobId": 28237, "location": "AB_100-01-02-003-04W5"},
        {"jobId": 28238, "location": "AB_100-02-03-045-12W5"},
        {"jobId": 28239, "location": "AB_100-03-04-056-08W5"},
    ]


@pytest.fixture
def sample_guidehawk_data() -> List[Dict]:
    """Sample Guidehawk sensor data."""
    return [
        {
            "kdatetime": "2024-01-15T10:00:00",
            "jobid": "GH-12345",
            "bhaloc": "a",
            "annp": 15.5,
            "temp": 25.3,
            "wellname": "Test Well",
            "formation": "Viking"
        },
        {
            "kdatetime": "2024-01-15T10:00:01",
            "jobid": "GH-12345",
            "bhaloc": "a",
            "annp": 15.6,
            "temp": 25.4,
            "wellname": "Test Well",
            "formation": "Viking"
        }
    ]


@pytest.fixture
def sample_reliance_columns() -> str:
    """Sample Reliance column headers."""
    return "Date,Time,Selected_Pressure (KPAg),Selected_Flow_Rate (L/min),Injection_Flag,Falloff_Flag,Stage_Number"


@pytest.fixture
def sample_reliance_data() -> List[List]:
    """Sample Reliance data records."""
    return [
        ["15/01/2024", "10:00:00", "15500", "100", "1", "0", "1"],
        ["15/01/2024", "10:00:01", "15600", "100", "1", "0", "1"],
        ["15/01/2024", "10:00:02", "15700", "100", "1", "0", "1"],
    ]


@pytest.fixture
def sample_guidehawk_df(sample_guidehawk_data) -> pd.DataFrame:
    """Sample Guidehawk DataFrame."""
    return pd.DataFrame(sample_guidehawk_data)


@pytest.fixture
def sample_reliance_df(sample_reliance_data, sample_reliance_columns) -> pd.DataFrame:
    """Sample Reliance DataFrame."""
    columns = sample_reliance_columns.split(',')
    return pd.DataFrame(sample_reliance_data, columns=columns)


@pytest.fixture
def oauth_token() -> str:
    """Sample OAuth token."""
    return "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test_token"


@pytest.fixture
def processed_jobs() -> set:
    """Sample set of processed job IDs."""
    return {"28235", "28236", "28237"}


@pytest.fixture
async def mock_aiohttp_session() -> AsyncMock:
    """Mock aiohttp ClientSession."""
    session = AsyncMock(spec=aiohttp.ClientSession)
    
    # Mock response
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json = AsyncMock(return_value={"result": []})
    
    # Mock context manager
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)
    
    session.get = Mock(return_value=mock_response)
    session.post = Mock(return_value=mock_response)
    
    return session


@pytest.fixture
def mock_settings(monkeypatch):
    """Mock settings for testing."""
    monkeypatch.setenv("EMAIL_SMTP_HOST", "smtp.test.com")
    monkeypatch.setenv("EMAIL_SMTP_PORT", "587")
    monkeypatch.setenv("EMAIL_FROM_ADDRESS", "test@test.com")
    monkeypatch.setenv("EMAIL_USERNAME", "test_user")
    monkeypatch.setenv("EMAIL_PASSWORD", "test_pass")
    monkeypatch.setenv("EMAIL_RECIPIENT", "recipient@test.com")
    
    monkeypatch.setenv("RELIANCE_API_BASE_URL", "https://test.reliance.com")
    monkeypatch.setenv("RELIANCE_OAUTH_URL", "https://test.reliance.com/oauth/token")
    monkeypatch.setenv("RELIANCE_CLIENT_ID", "test_client")
    monkeypatch.setenv("RELIANCE_CLIENT_SECRET", "test_secret")
    
    monkeypatch.setenv("KOBOLD_API_BASE_URL", "https://test.kobold.com")
    monkeypatch.setenv("HTTP_TIMEOUT", "30")


@pytest.fixture
def temp_csv_file(tmp_path):
    """Create a temporary CSV file for testing."""
    csv_file = tmp_path / "test_report.csv"
    csv_file.write_text("UWI,Pressure,Temperature\n100/01-02-003-04W5,1500,25\n")
    return str(csv_file)


@pytest.fixture
def sample_uwi_data() -> tuple:
    """Sample UWI parsing data."""
    return (
        "100/01-02-003-04W5",  # uwi
        "003",  # township
        "04"   # range
    )


# Async test support
@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()