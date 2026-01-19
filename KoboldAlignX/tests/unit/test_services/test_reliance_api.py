"""
Unit tests for reliance_api.py
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
import aiohttp
import pandas as pd

from src.services.reliance_api import RelianceAPIClient


class TestRelianceAPIClient:
    """Test suite for RelianceAPIClient."""
    
    @pytest.mark.asyncio
    async def test_get_oauth_token_success(self, mock_aiohttp_session):
        """Test successful OAuth token retrieval."""
        # Setup mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"access_token": "test_token_12345"})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        mock_aiohttp_session.post = Mock(return_value=mock_response)
        
        # Test
        client = RelianceAPIClient(mock_aiohttp_session)
        token = await client.get_oauth_token()
        
        # Verify
        assert token == "test_token_12345"
        assert client._access_token == "test_token_12345"
        mock_aiohttp_session.post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_oauth_token_failure(self, mock_aiohttp_session):
        """Test OAuth token retrieval failure."""
        # Setup mock response
        mock_response = AsyncMock()
        mock_response.status = 401
        mock_response.json = AsyncMock(return_value={"error_description": "Invalid credentials"})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        mock_aiohttp_session.post = Mock(return_value=mock_response)
        
        # Test
        client = RelianceAPIClient(mock_aiohttp_session)
        token = await client.get_oauth_token()
        
        # Verify
        assert token is None
        assert client._access_token is None
    
    @pytest.mark.asyncio
    async def test_get_jobs_success(self, mock_aiohttp_session, sample_jobs):
        """Test successful job retrieval."""
        # Setup mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"result": sample_jobs})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        mock_aiohttp_session.get = Mock(return_value=mock_response)
        
        # Test
        client = RelianceAPIClient(mock_aiohttp_session)
        jobs = await client.get_jobs("test_token")
        
        # Verify
        assert len(jobs) == 3
        assert jobs[0]["jobId"] == 28237
        assert jobs[1]["jobId"] == 28238
        mock_aiohttp_session.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_jobs_empty_result(self, mock_aiohttp_session):
        """Test getting jobs when result is empty."""
        # Setup mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"result": []})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        mock_aiohttp_session.get = Mock(return_value=mock_response)
        
        # Test
        client = RelianceAPIClient(mock_aiohttp_session)
        jobs = await client.get_jobs("test_token")
        
        # Verify
        assert jobs == []
    
    @pytest.mark.asyncio
    async def test_get_jobs_uses_cached_token(self, mock_aiohttp_session, sample_jobs):
        """Test that get_jobs uses cached token if not provided."""
        # Setup mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"result": sample_jobs})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        mock_aiohttp_session.get = Mock(return_value=mock_response)
        
        # Test
        client = RelianceAPIClient(mock_aiohttp_session)
        client._access_token = "cached_token"
        jobs = await client.get_jobs()  # No token provided
        
        # Verify - should use cached token
        assert len(jobs) == 3
    
    @pytest.mark.asyncio
    async def test_get_reliance_columns_success(self, mock_aiohttp_session, sample_reliance_columns):
        """Test successful column retrieval."""
        # Setup mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "result": [{"header": sample_reliance_columns}]
        })
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        mock_aiohttp_session.get = Mock(return_value=mock_response)
        
        # Test
        client = RelianceAPIClient(mock_aiohttp_session)
        columns = await client.get_reliance_columns("test_token", "28238")
        
        # Verify
        assert columns == sample_reliance_columns
        assert "Date" in columns
        assert "Pressure" in columns
    
    @pytest.mark.asyncio
    async def test_get_reliance_columns_failure(self, mock_aiohttp_session):
        """Test column retrieval failure."""
        # Setup mock response
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        mock_aiohttp_session.get = Mock(return_value=mock_response)
        
        # Test
        client = RelianceAPIClient(mock_aiohttp_session)
        columns = await client.get_reliance_columns("test_token", "99999")
        
        # Verify
        assert columns is None
    
    @pytest.mark.asyncio
    async def test_get_job_data_success(
        self, 
        mock_aiohttp_session,
        sample_reliance_data,
        sample_reliance_columns,
        sample_guidehawk_data
    ):
        """Test successful job data retrieval."""
        # Setup mock responses for both APIs
        # Reliance API response
        reliance_response = AsyncMock()
        reliance_response.status = 200
        reliance_response.json = AsyncMock(return_value={
            "result": [{
                "records": [
                    {"values": row} for row in sample_reliance_data
                ]
            }]
        })
        reliance_response.__aenter__ = AsyncMock(return_value=reliance_response)
        reliance_response.__aexit__ = AsyncMock(return_value=None)
        
        # Guidehawk API response
        guidehawk_response = AsyncMock()
        guidehawk_response.status = 200
        guidehawk_response.json = AsyncMock(return_value=sample_guidehawk_data)
        guidehawk_response.__aenter__ = AsyncMock(return_value=guidehawk_response)
        guidehawk_response.__aexit__ = AsyncMock(return_value=None)
        
        # Mock session to return different responses for different URLs
        async def mock_get(url, **kwargs):
            if "guidehawkdatauwi" in url:
                return guidehawk_response
            else:
                return reliance_response
        
        mock_aiohttp_session.get = mock_get
        
        # Test
        client = RelianceAPIClient(mock_aiohttp_session)
        columns = sample_reliance_columns.split(',')
        guidehawk_df, reliance_df = await client.get_job_data(
            "test_token",
            "28238",
            columns,
            "100/01-02-003-04W5"
        )
        
        # Verify
        assert guidehawk_df is not None
        assert reliance_df is not None
        assert isinstance(guidehawk_df, pd.DataFrame)
        assert isinstance(reliance_df, pd.DataFrame)
        assert len(guidehawk_df) == 2
        assert len(reliance_df) == 3
    
    @pytest.mark.asyncio
    async def test_get_job_data_guidehawk_failure(
        self,
        mock_aiohttp_session,
        sample_reliance_data,
        sample_reliance_columns
    ):
        """Test job data retrieval when Guidehawk API fails."""
        # Reliance succeeds
        reliance_response = AsyncMock()
        reliance_response.status = 200
        reliance_response.json = AsyncMock(return_value={
            "result": [{
                "records": [{"values": row} for row in sample_reliance_data]
            }]
        })
        reliance_response.__aenter__ = AsyncMock(return_value=reliance_response)
        reliance_response.__aexit__ = AsyncMock(return_value=None)
        
        # Guidehawk fails
        guidehawk_response = AsyncMock()
        guidehawk_response.status = 500
        guidehawk_response.__aenter__ = AsyncMock(return_value=guidehawk_response)
        guidehawk_response.__aexit__ = AsyncMock(return_value=None)
        
        # Mock session
        async def mock_get(url, **kwargs):
            if "guidehawkdatauwi" in url:
                return guidehawk_response
            else:
                return reliance_response
        
        mock_aiohttp_session.get = mock_get
        
        # Test
        client = RelianceAPIClient(mock_aiohttp_session)
        columns = sample_reliance_columns.split(',')
        guidehawk_df, reliance_df = await client.get_job_data(
            "test_token",
            "28238",
            columns,
            "100/01-02-003-04W5"
        )
        
        # Verify - both should be None
        assert guidehawk_df is None
        assert reliance_df is None
    
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager functionality."""
        async with RelianceAPIClient() as client:
            assert client._session is not None
        
        # Session should be closed after exiting context
    
    def test_initialization(self):
        """Test RelianceAPIClient initialization."""
        client = RelianceAPIClient()
        
        assert client.base_url is not None
        assert client.oauth_url is not None
        assert client.client_id is not None
        assert client.client_secret is not None
        assert client._session is None
        assert client._access_token is None
    
    @pytest.mark.asyncio
    async def test_initialization_with_session(self, mock_aiohttp_session):
        """Test initialization with provided session."""
        client = RelianceAPIClient(mock_aiohttp_session)
        
        assert client._session == mock_aiohttp_session


class TestRelianceAPIErrorHandling:
    """Test error handling in RelianceAPIClient."""
    
    @pytest.mark.asyncio
    async def test_network_error_get_jobs(self, mock_aiohttp_session):
        """Test handling of network error in get_jobs."""
        mock_aiohttp_session.get = Mock(side_effect=aiohttp.ClientError("Network error"))
        
        client = RelianceAPIClient(mock_aiohttp_session)
        jobs = await client.get_jobs("test_token")
        
        assert jobs is None
    
    @pytest.mark.asyncio
    async def test_unexpected_error_get_job_data(self, mock_aiohttp_session):
        """Test handling of unexpected error in get_job_data."""
        mock_aiohttp_session.get = Mock(side_effect=Exception("Unexpected error"))
        
        client = RelianceAPIClient(mock_aiohttp_session)
        guidehawk_df, reliance_df = await client.get_job_data(
            "test_token",
            "28238",
            ["col1", "col2"],
            "100/01-02-003-04W5"
        )
        
        assert guidehawk_df is None
        assert reliance_df is None