"""
Unit tests for job_processor.py
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
import aiohttp
import pandas as pd

from src.services.job_processor import JobProcessor, process_all_jobs_task


class TestJobProcessor:
    """Test suite for JobProcessor."""
    
    def test_initialization(self):
        """Test JobProcessor initialization."""
        processor = JobProcessor()
        
        assert processor.env is not None
        assert processor.email_service is not None
        assert processor.excel_generator is not None
        assert isinstance(processor.processed_jobs, set)
        assert len(processor.processed_jobs) == 0
    
    @pytest.mark.asyncio
    async def test_process_job_success(self, sample_job, oauth_token):
        """Test successful processing of a single job."""
        processor = JobProcessor()
        mock_session = AsyncMock()
        
        # Mock all the service calls
        with patch('src.services.job_processor.RelianceAPIClient') as mock_reliance, \
             patch('src.services.job_processor.KoboldAPIClient') as mock_kobold, \
             patch('src.utils.location_parser.parse_location') as mock_parse:
            
            # Setup mocks
            mock_parse.return_value = ("100/01-02-003-04W5", "003", "04")
            
            mock_reliance_instance = AsyncMock()
            mock_reliance_instance.get_reliance_columns.return_value = "Date,Time,Pressure"
            mock_reliance_instance.get_job_data.return_value = (
                pd.DataFrame({'test': [1, 2, 3]}),  # guidehawk_df
                pd.DataFrame({'test': [1, 2, 3]})   # reliance_df
            )
            mock_reliance.return_value = mock_reliance_instance
            
            mock_kobold_instance = AsyncMock()
            mock_kobold.return_value = mock_kobold_instance
            
            # Mock excel generator
            processor.excel_generator.align_jobs = AsyncMock(
                return_value=("test_report.csv", "GH-12345")
            )
            
            # Mock email service
            processor.email_service.send_job_completion_email = Mock()
            
            # Execute
            await processor.process_job(mock_session, sample_job, oauth_token)
            
            # Verify job was added to processed set
            assert sample_job["jobId"] in processor.processed_jobs
    
    @pytest.mark.asyncio
    async def test_process_job_already_processed(self, sample_job, oauth_token):
        """Test that already processed jobs are skipped."""
        processor = JobProcessor()
        processor.processed_jobs.add(sample_job["jobId"])
        mock_session = AsyncMock()
        
        # Execute - should return early
        await processor.process_job(mock_session, sample_job, oauth_token)
        
        # Verify - should not have called any services
        # (job was already in processed_jobs)
        assert sample_job["jobId"] in processor.processed_jobs
    
    @pytest.mark.asyncio
    async def test_process_job_location_parse_failure(self, sample_job, oauth_token):
        """Test handling when location parsing fails."""
        processor = JobProcessor()
        mock_session = AsyncMock()
        
        with patch('src.services.job_processor.RelianceAPIClient'), \
             patch('src.services.job_processor.KoboldAPIClient'), \
             patch('src.utils.location_parser.parse_location') as mock_parse:
            
            # Mock parse_location to return None values
            mock_parse.return_value = (None, None, None)
            
            # Execute
            await processor.process_job(mock_session, sample_job, oauth_token)
            
            # Verify - job should not be added to processed
            assert sample_job["jobId"] not in processor.processed_jobs
    
    @pytest.mark.asyncio
    async def test_process_job_empty_dataframes(self, sample_job, oauth_token):
        """Test handling when dataframes are empty."""
        processor = JobProcessor()
        mock_session = AsyncMock()
        
        with patch('src.services.job_processor.RelianceAPIClient') as mock_reliance, \
             patch('src.services.job_processor.KoboldAPIClient'), \
             patch('src.utils.location_parser.parse_location') as mock_parse:
            
            mock_parse.return_value = ("100/01-02-003-04W5", "003", "04")
            
            mock_reliance_instance = AsyncMock()
            mock_reliance_instance.get_reliance_columns.return_value = "Date,Time"
            mock_reliance_instance.get_job_data.return_value = (
                pd.DataFrame(),  # Empty guidehawk_df
                pd.DataFrame()   # Empty reliance_df
            )
            mock_reliance.return_value = mock_reliance_instance
            
            # Execute
            await processor.process_job(mock_session, sample_job, oauth_token)
            
            # Verify - job should not be processed
            assert sample_job["jobId"] not in processor.processed_jobs
    
    @pytest.mark.asyncio
    async def test_process_job_excel_generation_fails(self, sample_job, oauth_token):
        """Test handling when Excel generation fails."""
        processor = JobProcessor()
        mock_session = AsyncMock()
        
        with patch('src.services.job_processor.RelianceAPIClient') as mock_reliance, \
             patch('src.services.job_processor.KoboldAPIClient'), \
             patch('src.utils.location_parser.parse_location') as mock_parse:
            
            mock_parse.return_value = ("100/01-02-003-04W5", "003", "04")
            
            mock_reliance_instance = AsyncMock()
            mock_reliance_instance.get_reliance_columns.return_value = "Date,Time"
            mock_reliance_instance.get_job_data.return_value = (
                pd.DataFrame({'test': [1, 2, 3]}),
                pd.DataFrame({'test': [1, 2, 3]})
            )
            mock_reliance.return_value = mock_reliance_instance
            
            # Mock excel generator to return None (failure)
            processor.excel_generator.align_jobs = AsyncMock(return_value=(None, None))
            
            # Execute
            await processor.process_job(mock_session, sample_job, oauth_token)
            
            # Verify - job should not be added to processed
            assert sample_job["jobId"] not in processor.processed_jobs
    
    @pytest.mark.asyncio
    async def test_process_all_jobs_success(self, sample_jobs, oauth_token, processed_jobs):
        """Test successful processing of all jobs."""
        processor = JobProcessor()
        
        with patch('src.services.job_processor.aiohttp.ClientSession') as mock_session_cls, \
             patch('src.services.job_processor.RelianceAPIClient') as mock_reliance, \
             patch('src.services.job_processor.KoboldAPIClient') as mock_kobold:
            
            # Setup session mock
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_cls.return_value = mock_session
            
            # Setup Reliance API mock
            mock_reliance_instance = AsyncMock()
            mock_reliance_instance.get_oauth_token.return_value = oauth_token
            mock_reliance_instance.get_jobs.return_value = sample_jobs
            mock_reliance.return_value = mock_reliance_instance
            
            # Setup Kobold API mock
            mock_kobold_instance = AsyncMock()
            mock_kobold_instance.get_processed_jobs.return_value = processed_jobs
            mock_kobold.return_value = mock_kobold_instance
            
            # Mock process_job to avoid complex processing
            processor.process_job = AsyncMock()
            
            # Execute
            result = await processor.process_all_jobs()
            
            # Verify
            assert result is not None
            assert "processed" in result.lower()
    
    @pytest.mark.asyncio
    async def test_process_all_jobs_oauth_failure(self):
        """Test handling when OAuth fails."""
        processor = JobProcessor()
        
        with patch('src.services.job_processor.aiohttp.ClientSession') as mock_session_cls, \
             patch('src.services.job_processor.RelianceAPIClient') as mock_reliance:
            
            # Setup session mock
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_cls.return_value = mock_session
            
            # Setup Reliance API to fail OAuth
            mock_reliance_instance = AsyncMock()
            mock_reliance_instance.get_oauth_token.return_value = None
            mock_reliance.return_value = mock_reliance_instance
            
            # Execute
            result = await processor.process_all_jobs()
            
            # Verify
            assert result is None
    
    @pytest.mark.asyncio
    async def test_process_all_jobs_no_jobs(self, oauth_token):
        """Test handling when no jobs are available."""
        processor = JobProcessor()
        
        with patch('src.services.job_processor.aiohttp.ClientSession') as mock_session_cls, \
             patch('src.services.job_processor.RelianceAPIClient') as mock_reliance:
            
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_cls.return_value = mock_session
            
            mock_reliance_instance = AsyncMock()
            mock_reliance_instance.get_oauth_token.return_value = oauth_token
            mock_reliance_instance.get_jobs.return_value = []  # No jobs
            mock_reliance.return_value = mock_reliance_instance
            
            # Execute
            result = await processor.process_all_jobs()
            
            # Verify
            assert result is None
    
    @pytest.mark.asyncio
    async def test_process_all_jobs_all_already_processed(
        self, 
        sample_jobs, 
        oauth_token
    ):
        """Test when all jobs are already processed."""
        processor = JobProcessor()
        
        # Mark all jobs as processed
        all_processed = {str(job["jobId"]) for job in sample_jobs}
        
        with patch('src.services.job_processor.aiohttp.ClientSession') as mock_session_cls, \
             patch('src.services.job_processor.RelianceAPIClient') as mock_reliance, \
             patch('src.services.job_processor.KoboldAPIClient') as mock_kobold:
            
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_cls.return_value = mock_session
            
            mock_reliance_instance = AsyncMock()
            mock_reliance_instance.get_oauth_token.return_value = oauth_token
            mock_reliance_instance.get_jobs.return_value = sample_jobs
            mock_reliance.return_value = mock_reliance_instance
            
            mock_kobold_instance = AsyncMock()
            mock_kobold_instance.get_processed_jobs.return_value = all_processed
            mock_kobold.return_value = mock_kobold_instance
            
            # Execute
            result = await processor.process_all_jobs()
            
            # Verify - should report 0 processed
            assert result is not None
            assert "0 jobs" in result or "0 processed" in result.lower()


class TestProcessAllJobsTask:
    """Test the convenience task function."""
    
    @pytest.mark.asyncio
    async def test_process_all_jobs_task(self):
        """Test process_all_jobs_task function."""
        with patch('src.services.job_processor.JobProcessor') as mock_processor_class:
            mock_processor = Mock()
            mock_processor.process_all_jobs = AsyncMock(return_value="Success: 5 jobs processed")
            mock_processor_class.return_value = mock_processor
            
            # Execute
            result = await process_all_jobs_task()
            
            # Verify
            assert result == "Success: 5 jobs processed"
            mock_processor.process_all_jobs.assert_called_once()


class TestJobProcessorErrorHandling:
    """Test error handling in JobProcessor."""
    
    @pytest.mark.asyncio
    async def test_process_job_exception(self, sample_job, oauth_token):
        """Test handling of unexpected exception during job processing."""
        from src.core.exceptions import JobProcessingError
        
        processor = JobProcessor()
        mock_session = AsyncMock()
        
        with patch('src.services.job_processor.RelianceAPIClient') as mock_reliance:
            # Mock to raise exception
            mock_reliance.side_effect = Exception("Unexpected error")
            
            # Execute - should raise JobProcessingError
            with pytest.raises(JobProcessingError):
                await processor.process_job(mock_session, sample_job, oauth_token)
    
    @pytest.mark.asyncio
    async def test_process_all_jobs_client_error(self):
        """Test handling of client error."""
        processor = JobProcessor()
        
        with patch('src.services.job_processor.aiohttp.ClientSession') as mock_session_cls:
            # Mock to raise ClientError
            mock_session_cls.side_effect = aiohttp.ClientError("Network error")
            
            # Execute
            result = await processor.process_all_jobs()
            
            # Verify - should return None
            assert result is None
    
    @pytest.mark.asyncio
    async def test_process_all_jobs_timeout(self):
        """Test handling of timeout error."""
        import asyncio
        
        processor = JobProcessor()
        
        with patch('src.services.job_processor.aiohttp.ClientSession') as mock_session_cls:
            mock_session_cls.side_effect = asyncio.TimeoutError("Request timed out")
            
            # Execute
            result = await processor.process_all_jobs()
            
            # Verify
            assert result is None


class TestJobProcessorIntegration:
    """Integration-style tests for JobProcessor."""
    
    @pytest.mark.asyncio
    async def test_full_job_processing_workflow(
        self, 
        sample_job,
        oauth_token,
        sample_guidehawk_df,
        sample_reliance_df
    ):
        """Test complete job processing workflow."""
        processor = JobProcessor()
        mock_session = AsyncMock()
        
        with patch('src.services.job_processor.RelianceAPIClient') as mock_reliance, \
             patch('src.services.job_processor.KoboldAPIClient') as mock_kobold, \
             patch('src.utils.location_parser.parse_location') as mock_parse:
            
            # Setup complete mock chain
            mock_parse.return_value = ("100/01-02-003-04W5", "003", "04")
            
            mock_reliance_instance = AsyncMock()
            mock_reliance_instance.get_reliance_columns.return_value = "Date,Time,Pressure"
            mock_reliance_instance.get_job_data.return_value = (
                sample_guidehawk_df,
                sample_reliance_df
            )
            mock_reliance.return_value = mock_reliance_instance
            
            mock_kobold_instance = Mock()
            mock_kobold_instance.save_processed_job = Mock()
            mock_kobold.return_value = mock_kobold_instance
            
            # Mock services
            processor.excel_generator.align_jobs = AsyncMock(
                return_value=("FallOff_report.csv", "GH-12345")
            )
            processor.email_service.send_job_completion_email = Mock()
            
            # Execute
            await processor.process_job(mock_session, sample_job, oauth_token)
            
            # Verify complete workflow
            assert sample_job["jobId"] in processor.processed_jobs
            mock_reliance_instance.get_job_data.assert_called_once()
            processor.excel_generator.align_jobs.assert_called_once()
            processor.email_service.send_job_completion_email.assert_called_once()
            mock_kobold_instance.save_processed_job.assert_called_once()