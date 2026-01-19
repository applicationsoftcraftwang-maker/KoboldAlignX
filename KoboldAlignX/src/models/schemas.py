"""
Pydantic schemas for API validation.

TODO: Move from src/core/schemas.py and update to Pydantic v2
"""

from pydantic import BaseModel, Field


class JobCreate(BaseModel):
    """Schema for creating a job."""
    # TODO: Define fields
    pass


class JobResponse(BaseModel):
    """Schema for job response."""
    # TODO: Define fields
    pass
