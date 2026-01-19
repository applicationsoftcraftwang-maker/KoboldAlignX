"""
Job orchestrator for processing well data.

This module orchestrates the complete workflow:
1. Fetch pending jobs from Reliance API
2. Retrieve GuideHawk and Reliance data
3. Process and align datasets
4. Apply transformations
5. Export results
6. Send email notifications
"""
import logging
from typing import Optional, Tuple, Set, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import aiohttp

from api_clients import RelianceAPIClient, InternalAPIClient
from data_processor import DataProcessor
from data_transformer import DataTransformer
from email_service import EmailService
from utils import parse_location
from config import DEFAULT_TO_EMAIL

logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Constants
# ============================================================================

class ProcessingStatus(Enum):
    """Job processing status."""
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    ALREADY_PROCESSED = "already_processed"


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class JobContext:
    """Context for a single job being processed."""
    job_id: int
    location: str
    uwi: Optional[str] = None
    twp: Optional[str] = None
    rng: Optional[str] = None
    guidehawk_job_id: Optional[str] = None
    output_filename: Optional[str] = None
    
    def __str__(self) -> str:
        return f"Job {self.job_id} ({self.location})"


@dataclass
class ProcessingResult:
    """Result of processing a single job."""
    job_id: int
    status: ProcessingStatus
    message: str
    filename: Optional[str] = None
    guidehawk_job_id: Optional[str] = None
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        """Check if processing was successful."""
        return self.status == ProcessingStatus.SUCCESS


@dataclass
class ProcessingStats:
    """Statistics for batch processing."""
    total_jobs: int = 0
    processed: int = 0
    skipped: int = 0
    failed: int = 0
    already_processed: int = 0
    
    def add_result(self, result: ProcessingResult) -> None:
        """Update stats based on processing result."""
        if result.status == ProcessingStatus.SUCCESS:
            self.processed += 1
        elif result.status == ProcessingStatus.SKIPPED:
            self.skipped += 1
        elif result.status == ProcessingStatus.FAILED:
            self.failed += 1
        elif result.status == ProcessingStatus.ALREADY_PROCESSED:
            self.already_processed += 1
    
    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary for logging."""
        return {
            'total_jobs': self.total_jobs,
            'processed': self.processed,
            'skipped': self.skipped,
            'failed': self.failed,
            'already_processed': self.already_processed
        }


@dataclass
class DataFetchResult:
    """Result of fetching data for a job."""
    reliance_columns: Optional[List[str]] = None
    reliance_data: Optional[dict] = None
    guidehawk_data: Optional[dict] = None
    success: bool = False
    error_message: Optional[str] = None


@dataclass
class AlignmentResult:
    """Result of data alignment and processing."""
    guidehawk_top_df: Optional[pd.DataFrame] = None
    guidehawk_bottom_df: Optional[pd.DataFrame] = None
    reliance_df: Optional[pd.DataFrame] = None
    output_filename: Optional[str] = None
    guidehawk_job_id: Optional[str] = None
    success: bool = False
    error_message: Optional[str] = None


# ============================================================================
# Job Orchestrator
# ============================================================================

class JobOrchestrator:
    """
    Orchestrates the entire job processing workflow.
    
    """
    
    def __init__(
        self,
        data_processor: Optional[DataProcessor] = None,
        email_service: Optional[EmailService] = None
    ):
        self.data_processor = data_processor or DataProcessor()
        self.email_service = email_service or EmailService()
    
    # ========================================================================
    # Main Entry Point
    # ========================================================================
    
    async def process_all_jobs(self) -> Optional[str]:
        """ 
        Main entry point for processing all pending jobs.
        
        This method:
        1. Authenticates with Reliance API
        2. Fetches all pending jobs
        3. Gets list of already processed jobs
        4. Processes each new job
        5. Returns summary
        
        Returns:
            Success message with stats or None on error
        """
        stats = ProcessingStats()
        
        try:
            async with aiohttp.ClientSession() as session:
                # Initialize API clients
                reliance_client = RelianceAPIClient(session)
                internal_client = InternalAPIClient(session)
                
                # Authenticate
                if not await self._authenticate(reliance_client):
                    return None
                
                # Fetch jobs
                jobs = await self._fetch_jobs(reliance_client)
                if not jobs:
                    return None
                
                stats.total_jobs = len(jobs)
                logger.info(f"Found {stats.total_jobs} total jobs")
                
                # Get processed jobs
                processed_jobs = await self._get_processed_jobs(internal_client)
                if processed_jobs is None:
                    return None
                
                # Process each job
                for job in jobs:
                    result = await self._process_job(
                        job,
                        reliance_client,
                        internal_client,
                        processed_jobs
                    )
                    
                    stats.add_result(result)
                    
                    if result.success:
                        processed_jobs.add(str(result.job_id))
                
                # Log summary
                summary = self._create_summary(stats)
                logger.info(summary)
                
                return summary
                
        except Exception as e:
            logger.error(f"Error in process_all_jobs: {str(e)}")
            return None
    
    # ========================================================================
    # Authentication & Setup
    # ========================================================================
    
    async def _authenticate(self, reliance_client: RelianceAPIClient) -> bool:
        """ Authenticate with Reliance API."""
        logger.info("Authenticating with Reliance API...")
        
        token = await reliance_client.get_oauth_token()
        if not token:
            logger.error("Failed to obtain OAuth token")
            return False
        
        logger.info("OAuth token obtained successfully")
        return True
    
    async def _fetch_jobs(self, reliance_client: RelianceAPIClient) -> Optional[List[dict]]:
        """ Fetch all jobs from Reliance API. """
        logger.info("Fetching jobs from Reliance API...")
        
        jobs = await reliance_client.get_jobs()
        if not jobs:
            logger.warning("No jobs found")
            return None
        
        logger.info(f"Fetched {len(jobs)} jobs")
        return jobs
    
    async def _get_processed_jobs(self, internal_client: InternalAPIClient) -> Optional[Set[str]]:
        """ Get set of already processed job IDs."""
        logger.info("Fetching list of processed jobs...")
        
        processed_jobs = await internal_client.get_processed_jobs()
        if processed_jobs is None:
            logger.error("Failed to fetch processed jobs list")
            return None
        
        logger.info(f"Found {len(processed_jobs)} already processed jobs")
        return processed_jobs
    
    # ========================================================================
    # Job Processing
    # ========================================================================
    
    async def _process_job(
        self,
        job: dict,
        reliance_client: RelianceAPIClient,
        internal_client: InternalAPIClient,
        processed_jobs: Set[str]
    ) -> ProcessingResult:
        """ Process a single job (wrapper with error handling). """
        job_id = job["jobId"]
        
        # Check if already processed
        if str(job_id) in processed_jobs:
            logger.info(f"Job {job_id} already processed, skipping")
            return ProcessingResult(
                job_id=job_id,
                status=ProcessingStatus.ALREADY_PROCESSED,
                message=f"Job {job_id} already processed"
            )
        
        # Process the job
        try:
            return await self._process_single_job(
                job,
                reliance_client,
                internal_client
            )
        except Exception as e:
            logger.error(f"Unexpected error processing job {job_id}: {str(e)}")
            return ProcessingResult(
                job_id=job_id,
                status=ProcessingStatus.FAILED,
                message=f"Unexpected error: {str(e)}",
                error=str(e)
            )
    
    async def _process_single_job(
        self,
        job: dict,
        reliance_client: RelianceAPIClient,
        internal_client: InternalAPIClient
    ) -> ProcessingResult:
        """ Process a single job through the complete workflow. """
        # Create job context
        context = JobContext(
            job_id=job["jobId"],
            location=job["location"]
        )
        
        logger.info(f"Processing {context}")
        
        # Step 1: Parse location
        if not self._parse_job_location(context):
            return ProcessingResult(
                job_id=context.job_id,
                status=ProcessingStatus.SKIPPED,
                message=f"Failed to parse location: {context.location}"
            )
        
        # Step 2: Fetch data
        data_result = await self._fetch_job_data(
            context,
            reliance_client,
            internal_client
        )
        
        if not data_result.success:
            return ProcessingResult(
                job_id=context.job_id,
                status=ProcessingStatus.FAILED,
                message=data_result.error_message or "Failed to fetch data"
            )
        
        # Step 3: Process and align data
        alignment_result = await self._align_and_export_data(
            context,
            data_result
        )
        
        if not alignment_result.success:
            return ProcessingResult(
                job_id=context.job_id,
                status=ProcessingStatus.FAILED,
                message=alignment_result.error_message or "Failed to align data"
            )
        
        # Update context with results
        context.output_filename = alignment_result.output_filename
        context.guidehawk_job_id = alignment_result.guidehawk_job_id
        
        # Step 4: Send email notification
        if not self._send_email_notification(context):
            return ProcessingResult(
                job_id=context.job_id,
                status=ProcessingStatus.FAILED,
                message="Failed to send email notification",
                filename=context.output_filename,
                guidehawk_job_id=context.guidehawk_job_id
            )
        
        # Step 5: Save as processed
        self._save_processed_job(context, internal_client)
        
        # Return success
        logger.info(f"Successfully processed {context}")
        return ProcessingResult(
            job_id=context.job_id,
            status=ProcessingStatus.SUCCESS,
            message=f"Successfully processed {context}",
            filename=context.output_filename,
            guidehawk_job_id=context.guidehawk_job_id
        )
    
    # ========================================================================
    # Processing Steps
    # ========================================================================
    
    def _parse_job_location(self, context: JobContext) -> bool:
        """ Parse job location into UWI, township, and range."""
        logger.debug(f"Parsing location: {context.location}")
        
        uwi, twp, rng = parse_location(context.location)
        
        if not uwi or not twp or not rng:
            logger.warning(f"Failed to parse location for job {context.job_id}")
            return False
        
        context.uwi = uwi
        context.twp = twp
        context.rng = rng
        
        logger.info(f"Parsed location: uwi={uwi}, twp={twp}, rng={rng}")
        return True
    
    async def _fetch_job_data(
        self,
        context: JobContext,
        reliance_client: RelianceAPIClient,
        internal_client: InternalAPIClient
    ) -> DataFetchResult:
        """ Fetch all required data for job processing."""
        result = DataFetchResult()
        
        # Fetch Reliance job info
        logger.debug(f"Fetching Reliance job info for {context.job_id}")
        result.reliance_columns = await reliance_client.get_job_info(context.job_id)
        if not result.reliance_columns:
            result.error_message = f"Failed to fetch job info for {context.job_id}"
            return result
        
        # Fetch Reliance data
        logger.debug(f"Fetching Reliance data for {context.job_id}")
        result.reliance_data = await reliance_client.get_job_data(context.job_id)
        if not result.reliance_data:
            result.error_message = f"Failed to fetch Reliance data for {context.job_id}"
            return result
        
        # Fetch GuideHawk data
        logger.debug(f"Fetching GuideHawk data for {context.uwi}")
        result.guidehawk_data = await internal_client.get_guidehawk_data(context.uwi)
        if not result.guidehawk_data:
            result.error_message = f"Failed to fetch GuideHawk data for {context.uwi}"
            return result
        
        result.success = True
        logger.info(f"Successfully fetched all data for {context}")
        return result
    
    async def _align_and_export_data(
        self,
        context: JobContext,
        data: DataFetchResult
    ) -> AlignmentResult:
        """ Process, align, transform, and export data. """
        result = AlignmentResult()
        
        try:
            # Step 1: Prepare data
            preparation_result = self._prepare_data(data)
            if not preparation_result:
                result.error_message = "Failed to prepare data"
                return result
            
            gh_top, gh_bottom, rel_df = preparation_result
            
            # Step 2: Align datasets
            gh_top, gh_bottom, rel_df = self._align_datasets(gh_top, gh_bottom, rel_df)
            if gh_top is None or rel_df is None:
                result.error_message = "Failed to align datasets"
                return result
            
            # Step 3: Trim to peaks
            gh_top, gh_bottom, rel_df = self._trim_datasets(gh_top, gh_bottom, rel_df)
            
            # Step 4: Synchronize timestamps
            guidehawk_job_id = self._synchronize_timestamps(gh_top, gh_bottom, rel_df)
            
            # Step 5: Apply transformations
            gh_top, gh_bottom, rel_df, tundra_uwi = self._apply_transformations(
                gh_top,
                gh_bottom,
                rel_df,
                context
            )
            
            # Step 6: Export
            filename = self._export_data(gh_top, gh_bottom, rel_df, tundra_uwi)
            if not filename:
                result.error_message = "Failed to export data"
                return result
            
            # Success
            result.guidehawk_top_df = gh_top
            result.guidehawk_bottom_df = gh_bottom
            result.reliance_df = rel_df
            result.output_filename = filename
            result.guidehawk_job_id = guidehawk_job_id
            result.success = True
            
            logger.info(f"Data processing complete for {context}")
            return result
            
        except Exception as e:
            logger.error(f"Error in align_and_export_data: {str(e)}")
            result.error_message = str(e)
            return result
    
    def _prepare_data(
        self,
        data: DataFetchResult
    ) -> Optional[Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]]:
        """ Prepare GuideHawk and Reliance data."""
        logger.debug("Preparing data...")
        
        # Prepare GuideHawk data
        gh_top, gh_bottom = self.data_processor.prepare_guidehawk_data(data.guidehawk_data)
        if gh_top is None:
            logger.error("Failed to prepare GuideHawk data")
            return None
        
        # Prepare Reliance data
        rel_df = self.data_processor.prepare_reliance_data(
            data.reliance_data,
            data.reliance_columns
        )
        if rel_df is None:
            logger.error("Failed to prepare Reliance data")
            return None
        
        logger.info("Data preparation complete")
        return gh_top, gh_bottom, rel_df
    
    def _align_datasets(
        self,
        gh_top: pd.DataFrame,
        gh_bottom: Optional[pd.DataFrame],
        rel_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
        """ Align GuideHawk and Reliance datasets. """
        logger.debug("Aligning datasets...")
        
        # Align GuideHawk top/bottom if bottom exists
        if gh_bottom is not None and not gh_bottom.empty:
            gh_top, gh_bottom = self.data_processor.align_guidehawk_datasets(
                gh_top,
                gh_bottom
            )
            logger.info("GuideHawk datasets aligned")
        
        # Align GuideHawk with Reliance
        gh_top, gh_bottom, rel_df = self.data_processor.align_guidehawk_with_reliance(
            gh_top,
            gh_bottom if gh_bottom is not None else pd.DataFrame(),
            rel_df
        )
        
        logger.info("GuideHawk and Reliance aligned")
        return gh_top, gh_bottom, rel_df
    
    def _trim_datasets(
        self,
        gh_top: pd.DataFrame,
        gh_bottom: Optional[pd.DataFrame],
        rel_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
        """ Trim datasets to peak pressure regions."""
        logger.debug("Trimming to peak regions...")
        
        gh_top, gh_bottom, rel_df = self.data_processor.trim_to_peaks(
            gh_top,
            gh_bottom if gh_bottom is not None else pd.DataFrame(),
            rel_df
        )
        
        logger.info("Data trimmed to peaks")
        return gh_top, gh_bottom, rel_df
    
    def _synchronize_timestamps(
        self,
        gh_top: pd.DataFrame,
        gh_bottom: Optional[pd.DataFrame],
        rel_df: pd.DataFrame
    ) -> str:
        """ Synchronize timestamps across datasets. """
        logger.debug("Synchronizing timestamps...")
        
        # Get GuideHawk job ID
        guidehawk_job_id = gh_top["jobid"].iloc[0]
        
        # Synchronize timestamps
        gh_top["Date/Time"] = rel_df["Date/Time"]
        
        if gh_bottom is not None and not gh_bottom.empty:
            gh_bottom["Date/Time"] = rel_df["Date/Time"]
        
        logger.info("Timestamps synchronized")
        return guidehawk_job_id
    
    def _apply_transformations(
        self,
        gh_top: pd.DataFrame,
        gh_bottom: Optional[pd.DataFrame],
        rel_df: pd.DataFrame,
        context: JobContext
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame, str]:
        """ Apply final transformations and calculations. """
        logger.debug("Applying transformations...")
        
        gh_top, gh_bottom, rel_df, tundra_uwi = DataTransformer.prepare_final_dataframes(
            gh_top,
            gh_bottom if gh_bottom is not None else pd.DataFrame(),
            rel_df,
            context.uwi,
            context.twp,
            context.rng
        )
        
        logger.info("Final transformations complete")
        return gh_top, gh_bottom, rel_df, tundra_uwi
    
    def _export_data(
        self,
        gh_top: pd.DataFrame,
        gh_bottom: Optional[pd.DataFrame],
        rel_df: pd.DataFrame,
        tundra_uwi: str
    ) -> Optional[str]:
        """ Merge and export data to CSV. """
        logger.debug("Exporting data...")
        
        filename = DataTransformer.merge_and_export(
            gh_top,
            gh_bottom,
            rel_df,
            tundra_uwi
        )
        
        if filename:
            logger.info(f"Data exported to {filename}")
        
        return filename
    
    # ========================================================================
    # Notification & Completion
    # ========================================================================
    
    def _send_email_notification(self, context: JobContext) -> bool:
        """ Send email notification for completed job. """
        logger.debug(f"Sending email notification for {context}")
        
        subject = f"FallOff_{context.uwi}"
        
        email_sent = self.email_service.send_report(
            subject=subject,
            uwi=context.uwi,
            reliance_job_id=str(context.job_id),
            guidehawk_job_id=str(context.guidehawk_job_id),
            attachment_path=context.output_filename,
            to_email=DEFAULT_TO_EMAIL
        )
        
        if email_sent:
            logger.info(f"Email sent for {context}")
        else:
            logger.error(f"Failed to send email for {context}")
        
        return email_sent
    
    def _save_processed_job(
        self,
        context: JobContext,
        internal_client: InternalAPIClient
    ) -> None:
        """
        Save job as processed in internal system.
        
        Args:
            context: Job context
            internal_client: Internal API client
        """
        logger.debug(f"Saving {context} as processed")
        
        internal_client.save_processed_job(
            kobold_job_id=context.guidehawk_job_id,
            reliance_job_id=context.job_id,
            uwi=context.uwi
        )
        
        logger.info(f"Saved {context} as processed")
    
    # ========================================================================
    # Summary & Reporting
    # ========================================================================
    
    def _create_summary(self, stats: ProcessingStats) -> str:
        """ Create processing summary message. """
        summary_parts = [
            f"Processing completed:",
            f"  Total jobs: {stats.total_jobs}",
            f"  Processed: {stats.processed}",
            f"  Already processed: {stats.already_processed}",
            f"  Skipped: {stats.skipped}",
            f"  Failed: {stats.failed}"
        ]
        
        return "\n".join(summary_parts)


async def process_jobs(
    data_processor: Optional[DataProcessor] = None,
    email_service: Optional[EmailService] = None
) -> Optional[str]:
    """ Convenience function to process all jobs. """
    orchestrator = JobOrchestrator(
        data_processor=data_processor,
        email_service=email_service
    )
    
    return await orchestrator.process_all_jobs()