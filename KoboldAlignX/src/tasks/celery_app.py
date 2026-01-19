"""
Celery application configuration.

TODO: Extract from celery_worker.py lines 33-54
"""

import os
from celery import Celery
from celery.schedules import timedelta

from src.core.config import settings

# Create Celery application
celery = Celery(__name__)

celery.conf.update(
    broker_url=settings.celery_broker_url,
    result_backend=settings.celery_result_backend,
    beat_scheduler='celery.beat.PersistentScheduler',
    timezone=settings.celery_timezone,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_reject_on_worker_lost=True,
    broker_transport_options={'confirm_publish': True},
    task_time_limit=settings.celery_task_time_limit,
    task_soft_time_limit=settings.celery_task_soft_time_limit,
    beat_max_loop_interval=1500
)

# TODO: Configure beat schedule
celery.conf.beat_schedule = {
    "run-every-15-minutes": {
        "task": "create_task",
        "schedule": timedelta(minutes=settings.celery_beat_schedule_interval),
        "args": (0, 0, 26998),
    },
}
