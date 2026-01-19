"""Main Celery application and task definitions."""
import asyncio
import logging
from celery import Celery
from celery.schedules import timedelta
from celery.utils.log import get_task_logger

from config import (
    CELERY_BROKER_URL,
    CELERY_RESULT_BACKEND,
    CELERY_TIMEZONE,
    CELERY_TASK_TIME_LIMIT,
    CELERY_TASK_SOFT_TIME_LIMIT,
    CELERY_BEAT_MAX_LOOP_INTERVAL,
    CELERY_BEAT_SCHEDULE_INTERVAL_MINUTES
)
from job_orchestrator import JobOrchestrator

# Setup logger
logger = get_task_logger(__name__)

# Initialize Celery application
celery = Celery(__name__)
celery.conf.update(
    broker_url=CELERY_BROKER_URL,
    result_backend=CELERY_RESULT_BACKEND,
    beat_scheduler='celery.beat.PersistentScheduler',
    timezone=CELERY_TIMEZONE,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_reject_on_worker_lost=True,
    broker_transport_options={'confirm_publish': True},
    task_time_limit=CELERY_TASK_TIME_LIMIT,
    task_soft_time_limit=CELERY_TASK_SOFT_TIME_LIMIT,
    beat_max_loop_interval=CELERY_BEAT_MAX_LOOP_INTERVAL
)

# Configure periodic task schedule (same pattern as before)
celery.conf.beat_schedule = {
    "run-every-15-minutes": {
        "task": "create_task",
        "schedule": timedelta(minutes=CELERY_BEAT_SCHEDULE_INTERVAL_MINUTES),
        "args": (0, 0, 0),  # Placeholder args for compatibility
    },
}


@celery.task(name="create_task", bind=True, max_retries=3)
def create_task(self, a=0, b=0, job_id=0):
    """ Main Celery task to process all pending jobs.
    
    This task maintains the same interface as the original for compatibility.
    The parameters a, b, job_id are kept for backward compatibility but not used.
    
    """
    try:
        logger.info("Starting job processing task")
        logger.info(f"Task called with args: a={a}, b={b}, job_id={job_id}")
        
        orchestrator = JobOrchestrator()
        result = asyncio.run(orchestrator.process_all_jobs())
        
        if result:
            logger.info(f"Job processing completed: {result}")
            return result
        else:
            logger.warning("Job processing returned no result")
            return None
            
    except Exception as e:
        logger.error(f"Error in create_task: {str(e)}")
        
        # Retry with exponential backoff
        try:
            raise self.retry(exc=e, countdown=60 * (2 ** self.request.retries))
        except self.MaxRetriesExceededError:
            logger.error(f"Max retries exceeded for task {self.request.id}")
            return None


@celery.task(name="process_specific_job", bind=True, max_retries=3)
def process_specific_job(self, job_id: int):
    """ Process a specific job by Reliance job ID. """
    try:
        logger.info(f"Processing specific job: {job_id}")
        
        # For now, processes all jobs but could be modified to filter by job_id
        # You can customize the JobOrchestrator to accept a specific job_id
        orchestrator = JobOrchestrator()
        result = asyncio.run(orchestrator.process_all_jobs())
        
        if result:
            logger.info(f"Job {job_id} processing completed")
            return f"Job {job_id} processed successfully"
        else:
            logger.warning(f"Job {job_id} processing returned no result")
            return None
            
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {str(e)}")
        
        # Retry with exponential backoff
        try:
            raise self.retry(exc=e, countdown=60 * (2 ** self.request.retries))
        except self.MaxRetriesExceededError:
            logger.error(f"Max retries exceeded for job {job_id}")
            return None


if __name__ == "__main__":
    # For testing: run the task directly
    print("Running task manually...")
    result = create_task(0, 0, 0)
    print(f"Result: {result}")