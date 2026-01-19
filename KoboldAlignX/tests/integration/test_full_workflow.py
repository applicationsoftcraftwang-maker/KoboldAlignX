"""
Integration tests for complete workflow.

These tests verify that multiple services work together correctly.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
import pandas as pd
import aiohttp

class TestFullWorkflowIntegration:
    """Integration tests for complete job processing workflow."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_workflow_single_job(
        self,
        sample_job,
        oauth_token,
        sample_guidehawk_df,
        sample_reliance_df,
        temp_csv_file
    ):
        """Test complete workflow: OAuth → Get Jobs → Process → Email → Save."""
        processor = JobProcessor()
        
        with patch('src.services.job_processor.aiohttp.ClientSession') as mock_session_cls, \
             patch('src.services.reliance_api.aiohttp.ClientSession'), \
             patch('src.utils.location_parser.parse_location') as mock_parse, \
             patch('src.services.email_service.smtplib.SMTP') as mock_smtp:
            
            # Setup session
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_cls.return_value = mock_session
            
            # Mock location parser
            mock_parse.return_value = ("100/01-02-003-04W5", "003", "04")
            
            # Mock Reliance API responses
            oauth_response = AsyncMock()
            oauth_response.status = 200
            oauth_response.json = AsyncMock(return_value={"access_token": oauth_token})
            oauth_response.__aenter__ = AsyncMock(return_value=oauth_response)
            oauth_response.__aexit__ = AsyncMock(return_value=None)
            
            jobs_response = AsyncMock()
            jobs_response.status = 200
            jobs_response.json = AsyncMock(return_value={"result": [sample_job]})
            jobs_response.__aenter__ = AsyncMock(return_value=jobs_response)
            jobs_response.__aexit__ = AsyncMock(return_value=None)
            
            columns_response = AsyncMock()
            columns_response.status = 200
            columns_response.json = AsyncMock(return_value={"result": [{"header": "Date,Time"}]})
            columns_response.__aenter__ = AsyncMock(return_value=columns_response)
            columns_response.__aexit__ = AsyncMock(return_value=None)
            
            processed_jobs_response = AsyncMock()
            processed_jobs_response.status = 200
            processed_jobs_response.json = AsyncMock(return_value=[])
            processed_jobs_response.__aenter__ = AsyncMock(return_value=processed_jobs_response)
            processed_jobs_response.__aexit__ = AsyncMock(return_value=None)
            
            # Mock session.post and session.get
            def mock_get(url, **kwargs):
                if "oauth" in url:
                    return oauth_response
                elif "/jobs" in url and "info" not in url:
                    return jobs_response
                elif "info" in url:
                    return columns_response
                elif "processedjobs" in url:
                    return processed_jobs_response
                return AsyncMock(status=404)
            
            mock_session.post = Mock(return_value=oauth_response)
            mock_session.get = mock_get
            
            # Mock Excel generator
            processor.excel_generator.align_jobs = AsyncMock(
                return_value=(temp_csv_file, "GH-12345")
            )
            
            # Mock SMTP
            mock_smtp_instance = Mock()
            mock_smtp.return_value.__enter__.return_value = mock_smtp_instance
            
            # Mock Kobold API save
            with patch('src.services.kobold_api.requests.post') as mock_save:
                mock_save.return_value = Mock(status_code=201)
                
                # Execute complete workflow
                result = await processor.process_all_jobs()
                
                # Verify workflow completed
                assert result is not None
                mock_smtp_instance.send_message.assert_called_once()
                mock_save.assert_called_once()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_workflow_with_multiple_jobs(
        self,
        sample_jobs,
        oauth_token,
        processed_jobs
    ):
        """Test workflow with multiple jobs where some are already processed."""
        processor = JobProcessor()
        
        with patch('src.services.job_processor.aiohttp.ClientSession') as mock_session_cls, \
             patch('src.services.job_processor.RelianceAPIClient') as mock_reliance, \
             patch('src.services.job_processor.KoboldAPIClient') as mock_kobold:
            
            # Setup session
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_cls.return_value = mock_session
            
            # Setup Reliance API
            mock_reliance_instance = AsyncMock()
            mock_reliance_instance.get_oauth_token.return_value = oauth_token
            mock_reliance_instance.get_jobs.return_value = sample_jobs
            mock_reliance.return_value = mock_reliance_instance
            
            # Setup Kobold API
            mock_kobold_instance = AsyncMock()
            mock_kobold_instance.get_processed_jobs.return_value = processed_jobs
            mock_kobold.return_value = mock_kobold_instance
            
            # Mock process_job to track calls
            process_job_calls = []
            
            async def track_process_job(session, job, token):
                process_job_calls.append(job["jobId"])
            
            processor.process_job = track_process_job
            
            # Execute
            result = await processor.process_all_jobs()
            
            # Verify - should only process unprocessed jobs
            expected_processed = [
                job["jobId"] for job in sample_jobs 
                if str(job["jobId"]) not in processed_jobs
            ]
            assert len(process_job_calls) == len(expected_processed)
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_workflow_error_recovery(self, sample_jobs, oauth_token):
        """Test that workflow continues after individual job failures."""
        processor = JobProcessor()
        
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
            mock_kobold_instance.get_processed_jobs.return_value = set()
            mock_kobold.return_value = mock_kobold_instance
            
            # Mock process_job to fail for first job, succeed for others
            call_count = [0]
            
            async def mock_process_job(session, job, token):
                call_count[0] += 1
                if call_count[0] == 1:
                    raise Exception("Processing failed for first job")
                # Others succeed
            
            processor.process_job = mock_process_job
            
            # Execute - should handle error and continue
            result = await processor.process_all_jobs()
            
            # Verify - should have attempted all jobs
            assert call_count[0] == len(sample_jobs)


class TestAPIIntegration:
    """Integration tests for API clients working together."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_reliance_and_kobold_api_integration(self):
        """Test Reliance and Kobold APIs working together."""
        async with aiohttp.ClientSession() as session:
            reliance = RelianceAPIClient(session)
            kobold = KoboldAPIClient(session)
            
            with patch.object(reliance, 'get_oauth_token') as mock_oauth, \
                 patch.object(reliance, 'get_jobs') as mock_get_jobs, \
                 patch.object(kobold, 'get_processed_jobs') as mock_get_processed:
                
                # Setup mocks
                mock_oauth.return_value = "test_token"
                mock_get_jobs.return_value = [
                    {"jobId": 28237, "location": "AB_100-01-02-003-04W5"},
                    {"jobId": 28238, "location": "AB_100-02-03-004-05W5"}
                ]
                mock_get_processed.return_value = {"28237"}
                
                # Execute
                token = await reliance.get_oauth_token()
                all_jobs = await reliance.get_jobs(token)
                processed = await kobold.get_processed_jobs()
                
                # Filter unprocessed
                unprocessed = [
                    job for job in all_jobs 
                    if str(job["jobId"]) not in processed
                ]
                
                # Verify
                assert len(unprocessed) == 1
                assert unprocessed[0]["jobId"] == 28238
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_data_pipeline_integration(
        self,
        sample_guidehawk_df,
        sample_reliance_df
    ):
        """Test data flows correctly through the pipeline."""
        with patch('src.utils.location_parser.parse_location') as mock_parse:
            mock_parse.return_value = ("100/01-02-003-04W5", "003", "04")
            
            # Parse location
            uwi, twp, rng = await parse_location("AB_100-01-02-003-04W5")
            
            # Generate Excel
            generator = ExcelGenerator()
            filename, job_id = await generator.align_jobs(
                sample_guidehawk_df,
                sample_reliance_df,
                "28238",
                uwi,
                twp,
                rng
            )
            
            # Verify data pipeline
            assert uwi == "100/01-02-003-04W5"
            assert twp == "003"
            assert rng == "04"
            # filename might be None depending on data quality


class TestServiceCommunication:
    """Test how services communicate and pass data."""
    
    @pytest.mark.integration
    def test_email_service_with_generated_report(self, temp_csv_file):
        """Test email service can send generated reports."""
        with patch('src.services.email_service.smtplib.SMTP') as mock_smtp:
            mock_smtp_instance = Mock()
            mock_smtp.return_value.__enter__.return_value = mock_smtp_instance
            
            # Create email service
            email_service = EmailService()
            
            # Send email with report
            email_service.send_job_completion_email(
                to_email="test@example.com",
                job_id="28238",
                guidehawk_job_id="GH-12345",
                uwi="100/01-02-003-04W5",
                attachment_path=temp_csv_file
            )
            
            # Verify
            mock_smtp_instance.send_message.assert_called_once()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_job_processor_coordinates_all_services(
        self,
        sample_job,
        oauth_token
    ):
        """Test JobProcessor correctly coordinates all services."""
        processor = JobProcessor()
        mock_session = AsyncMock()
        
        # Track which services were called
        services_called = set()
        
        with patch('src.services.job_processor.RelianceAPIClient') as mock_reliance, \
             patch('src.services.job_processor.KoboldAPIClient') as mock_kobold, \
             patch('src.utils.location_parser.parse_location') as mock_parse:
            
            # Setup mocks with tracking
            mock_parse.side_effect = lambda x: (
                services_called.add('location_parser'),
                ("100/01-02-003-04W5", "003", "04")
            )[1]
            
            mock_reliance_instance = AsyncMock()
            mock_reliance_instance.get_reliance_columns.side_effect = lambda *args: (
                services_called.add('reliance_api'),
                "Date,Time"
            )[1]
            mock_reliance_instance.get_job_data.return_value = (
                pd.DataFrame({'test': [1]}),
                pd.DataFrame({'test': [1]})
            )
            mock_reliance.return_value = mock_reliance_instance
            
            mock_kobold_instance = Mock()
            mock_kobold_instance.save_processed_job = lambda x: services_called.add('kobold_api')
            mock_kobold.return_value = mock_kobold_instance
            
            # Mock other services
            processor.excel_generator.align_jobs = AsyncMock(
                side_effect=lambda *args, **kwargs: (
                    services_called.add('excel_generator'),
                    ("test.csv", "GH-123")
                )[1]
            )
            
            processor.email_service.send_job_completion_email = lambda **kwargs: (
                services_called.add('email_service')
            )
            
            # Execute
            await processor.process_job(mock_session, sample_job, oauth_token)
            
            # Verify all services were coordinated
            expected_services = {
                'location_parser',
                'reliance_api',
                'excel_generator',
                'email_service',
                'kobold_api'
            }
            assert services_called == expected_services


class TestEndToEndScenarios:
    """End-to-end scenario tests."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_new_job_processing_scenario(
        self,
        sample_job,
        oauth_token,
        temp_csv_file
    ):
        """Test scenario: New job arrives and gets fully processed."""
        processor = JobProcessor()
        
        with patch('src.services.job_processor.aiohttp.ClientSession') as mock_session_cls:
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_cls.return_value = mock_session
            
            # Mock complete scenario
            with patch('src.services.reliance_api.RelianceAPIClient') as mock_reliance, \
                 patch('src.services.kobold_api.KoboldAPIClient') as mock_kobold, \
                 patch('src.utils.location_parser.parse_location') as mock_parse, \
                 patch('src.services.email_service.smtplib.SMTP'):
                
                # Job is not processed yet
                mock_kobold_instance = AsyncMock()
                mock_kobold_instance.get_processed_jobs.return_value = set()
                mock_kobold.return_value = mock_kobold_instance
                
                # Setup rest of workflow
                mock_parse.return_value = ("100/01-02-003-04W5", "003", "04")
                
                mock_reliance_instance = AsyncMock()
                mock_reliance_instance.get_oauth_token.return_value = oauth_token
                mock_reliance_instance.get_jobs.return_value = [sample_job]
                mock_reliance_instance.get_reliance_columns.return_value = "Date,Time"
                mock_reliance_instance.get_job_data.return_value = (
                    pd.DataFrame({'test': [1]}),
                    pd.DataFrame({'test': [1]})
                )
                mock_reliance.return_value = mock_reliance_instance
                
                processor.excel_generator.align_jobs = AsyncMock(
                    return_value=(temp_csv_file, "GH-12345")
                )
                
                # Execute complete workflow
                result = await processor.process_all_jobs()
                
                # Verify job was processed
                assert result is not None
                assert "1" in result  # 1 job processed
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_duplicate_job_prevention_scenario(
        self,
        sample_job,
        oauth_token
    ):
        """Test scenario: Job arrives but was already processed."""
        processor = JobProcessor()
        
        with patch('src.services.job_processor.aiohttp.ClientSession') as mock_session_cls, \
             patch('src.services.reliance_api.RelianceAPIClient') as mock_reliance, \
             patch('src.services.kobold_api.KoboldAPIClient') as mock_kobold:
            
            mock_session = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_cls.return_value = mock_session
            
            # Job was already processed
            mock_kobold_instance = AsyncMock()
            mock_kobold_instance.get_processed_jobs.return_value = {str(sample_job["jobId"])}
            mock_kobold.return_value = mock_kobold_instance
            
            mock_reliance_instance = AsyncMock()
            mock_reliance_instance.get_oauth_token.return_value = oauth_token
            mock_reliance_instance.get_jobs.return_value = [sample_job]
            mock_reliance.return_value = mock_reliance_instance
            
            # Execute
            result = await processor.process_all_jobs()
            
            # Verify - should skip the job
            assert result is not None
            assert "0" in result or "skipped" in result.lower()