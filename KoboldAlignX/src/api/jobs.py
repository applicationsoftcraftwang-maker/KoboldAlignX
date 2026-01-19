"""
FastAPI application for managing Celery tasks.

This module provides REST API endpoints for creating and monitoring
asynchronous tasks using Celery.
"""

import os
from typing import Any, Dict

import uvicorn
from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from src.core.celery_worker import create_task

# Load environment variables
load_dotenv("src/core/.env")

# Initialize FastAPI app
app = FastAPI(
    title="Task Management API",
    description="API for managing asynchronous Celery tasks",
    version="1.0.0",
)


# Pydantic models for request/response validation
class TaskRequest(BaseModel):
    """Request model for creating a new task."""
    
    amount: int = Field(..., gt=0, description="Amount value (must be positive)")
    x: str = Field(..., min_length=1, description="X parameter")
    y: str = Field(..., min_length=1, description="Y parameter")

    class Config:
        schema_extra = {
            "example": {
                "amount": 100,
                "x": "example_x",
                "y": "example_y"
            }
        }


class TaskResponse(BaseModel):
    """Response model for task creation."""
    
    result: str = Field(..., description="Task result or token")
    task_id: str = Field(..., description="Celery task ID")


class TaskStatusResponse(BaseModel):
    """Response model for task status."""
    
    status: str = Field(..., description="Task status (pending, success, failure)")
    result: Any = Field(None, description="Task result if completed")
    error: str = Field(None, description="Error message if failed")


# API Endpoints
@app.get("/", tags=["Health"])
async def root() -> Dict[str, str]:
    """ Health check endpoint. """
    return {"message": "Task Management API is running"}


@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, str]:
   return {
        "status": "healthy",
        "service": "Task Management API",
        "version": "1.0.0"
    }


@app.post(
    "/reliancejob/{job_id}",
    response_model=TaskResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Tasks"]
)
async def create_reliance_job(
    job_id: int,
    task_data: TaskRequest = Body(...)
) -> TaskResponse:
    """
    Create and execute a new Celery task.
    """
    try:
        task = create_task.delay(task_data.amount, task_data.x, job_id)
        result = task.get(timeout=30)  
        
        return TaskResponse(
            result=result,
            task_id=task.id
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Task execution failed: {str(e)}"
        )


@app.get(
    "/task/{task_id}",
    response_model=TaskStatusResponse,
    tags=["Tasks"]
)
async def get_task_status(task_id: str) -> TaskStatusResponse:
    """ Retrieve the status and result of a task. """
    try:
        task = create_task.AsyncResult(task_id)
        
        if task.state == "PENDING":
            return TaskStatusResponse(status="pending")
        
        elif task.state == "FAILURE":
            return TaskStatusResponse(
                status="failure",
                error=str(task.result)
            )
        
        elif task.state == "SUCCESS":
            return TaskStatusResponse(
                status="success",
                result=task.result
            )
        
        else:
            return TaskStatusResponse(
                status=task.state.lower(),
                result=task.info
            )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid task ID or error retrieving task: {str(e)}"
        )


# Alternative async endpoint that doesn't block
@app.post(
    "/reliancejob/{job_id}/async",
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Tasks"]
)
async def create_reliance_job_async(
    job_id: int,
    task_data: TaskRequest = Body(...)
) -> Dict[str, str]:
    """ Create a new Celery task without waiting for completion. """
    try:
        task = create_task.delay(task_data.amount, task_data.x, job_id)
        
        return {
            "task_id": task.id,
            "status": "Task created successfully",
            "message": f"Use GET /task/{task.id} to check status"
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Task creation failed: {str(e)}"
        )


def main():
    """Run the FastAPI application."""
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()