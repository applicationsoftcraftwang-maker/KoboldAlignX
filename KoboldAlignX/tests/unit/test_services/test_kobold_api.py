"""
Unit tests for kobold_api.py
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
import aiohttp

from src.services.kobold_api import KoboldAPIClient


class TestKoboldAPIClient:
    """Test suite for KoboldAPIClient."""
    
    @pytest.mark.asyncio
    async def test_get_processed_jobs_success(self, mock_aiohttp_session):
        """Test successful retrieval of processed jobs."""
        # Setup mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=[
            {'reliance_job_id': '28237', 'kobold_job_id': 'GH-001', 'uwi': '100/01-02-003-04W5'},
            {'reliance_job_id': '28238', 'kobold_job_id': 'GH-002', 'uwi': '100/02-03-004-05W5'},
        ])
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        mock_aiohttp_session.get = Mock(return_value=mock_response)
        
        # Test
        client = KoboldAPIClient(mock_aiohttp_session)
        result = await client.get_processed_jobs()
        
        # Verify
        assert result == {'28237', '28238'}
        assert '28237' in result
        assert '28238' in result
        mock_aiohttp_session.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_processed_jobs_empty_list(self, mock_aiohttp_session):
        """Test getting processed jobs when list is empty."""
        # Setup mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=[])
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        mock_aiohttp_session.get = Mock(return_value=mock_response)
        
        # Test
        client = KoboldAPIClient(mock_aiohttp_session)
        result = await client.get_processed_jobs()
        
        # Verify
        assert result == set()
        assert len(result) == 0
    
    @pytest.mark.asyncio
    async def test_get_processed_jobs_api_error(self, mock_aiohttp_session):
        """Test handling of API error response."""
        # Setup mock response
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)
        
        mock_aiohttp_session.get = Mock(return_value=mock_response)
        
        # Test
        client = KoboldAPIClient(mock_aiohttp_session)
        result = await client.get_processed_jobs()
        
        # Verify
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_processed_jobs_client_error(self, mock_aiohttp_session):
        """Test handling of client error."""
        # Setup mock to raise exception
        mock_aiohttp_session.get = Mock(side_effect=aiohttp.ClientError("Connection error"))
        
        # Test
        client = KoboldAPIClient(mock_aiohttp_session)
        result = await client.get_processed_jobs()
        
        # Verify
        assert result is None
    
    @pytest.mark.asyncio
    async def test_get_processed_jobs_unexpected_error(self, mock_aiohttp_session):
        """Test handling of unexpected error."""
        # Setup mock to raise exception
        mock_aiohttp_session.get = Mock(side_effect=Exception("Unexpected error"))
        
        # Test
        client = KoboldAPIClient(mock_aiohttp_session)
        result = await client.get_processed_jobs()
        
        # Verify
        assert result is None
    
    @patch('src.services.kobold_api.requests.post')
    def test_save_processed_job_success(self, mock_post):
        """Test successful saving of processed job."""
        # Setup mock response
        mock_response = Mock()
        mock_response.status_code = 201
        mock_post.return_value = mock_response
        
        # Test
        client = KoboldAPIClient()
        job_data = {
            'kobold_job_id': 'GH-12345',
            'reliance_job_id': '28240',
            'uwi': '100/01-02-003-04W5'
        }
        
        # Should not raise exception
        client.save_processed_job(job_data)
        
        # Verify
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]['json'] == job_data
    
    @patch('src.services.kobold_api.requests.post')
    def test_save_processed_job_failure(self, mock_post):
        """Test handling of save failure."""
        # Setup mock response with error
        mock_response = Mock()
        mock_response.status_code = 400
        mock_post.return_value = mock_response
        
        # Test
        client = KoboldAPIClient()
        job_data = {
            'kobold_job_id': 'GH-12345',
            'reliance_job_id': '28240',
            'uwi': '100/01-02-003-04W5'
        }
        
        # Should not raise exception (logs error)
        client.save_processed_job(job_data)
        
        # Verify
        mock_post.assert_called_once()
    
    @patch('src.services.kobold_api.requests.post')
    def test_save_processed_job_request_exception(self, mock_post):
        """Test handling of request exception."""
        # Setup mock to raise exception
        import requests
        mock_post.side_effect = requests.RequestException("Network error")
        
        # Test
        client = KoboldAPIClient()
        job_data = {
            'kobold_job_id': 'GH-12345',
            'reliance_job_id': '28240',
            'uwi': '100/01-02-003-04W5'
        }
        
        # Should not raise exception (logs error)
        client.save_processed_job(job_data)
        
        # Verify
        mock_post.assert_called_once()
    
    def test_kobold_api_client_initialization(self):
        """Test KoboldAPIClient initialization."""
        client = KoboldAPIClient()
        
        assert client.base_url is not None
        assert client._session is None
    
    @pytest.mark.asyncio
    async def test_kobold_api_client_with_session(self, mock_aiohttp_session):
        """Test KoboldAPIClient with provided session."""
        client = KoboldAPIClient(mock_aiohttp_session)
        
        assert client._session == mock_aiohttp_session


class TestKoboldAPIIntegration:
    """Integration-style tests for KoboldAPIClient."""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self, mock_aiohttp_session):
        """Test complete workflow: get processed jobs, then save new one."""
        # Setup mock for get
        mock_get_response = AsyncMock()
        mock_get_response.status = 200
        mock_get_response.json = AsyncMock(return_value=[
            {'reliance_job_id': '28237', 'kobold_job_id': 'GH-001', 'uwi': '100/01-02-003-04W5'}
        ])
        mock_get_response.__aenter__ = AsyncMock(return_value=mock_get_response)
        mock_get_response.__aexit__ = AsyncMock(return_value=None)
        
        mock_aiohttp_session.get = Mock(return_value=mock_get_response)
        
        # Test getting processed jobs
        client = KoboldAPIClient(mock_aiohttp_session)
        processed = await client.get_processed_jobs()
        
        assert '28237' in processed
        assert '28238' not in processed
        
        # Test saving new job (with mocked requests.post)
        with patch('src.services.kobold_api.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 201
            mock_post.return_value = mock_response
            
            new_job = {
                'kobold_job_id': 'GH-002',
                'reliance_job_id': '28238',
                'uwi': '100/02-03-004-05W5'
            }
            client.save_processed_job(new_job)
            
            mock_post.assert_called_once()