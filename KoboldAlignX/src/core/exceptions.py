"""
Custom exception classes for KoboldAlignX application.

This module defines a hierarchy of custom exceptions that provide
clear error messages and proper error handling throughout the application.
"""

from typing import Any, Dict, Optional


class KoboldAlignXException(Exception):
    """Base exception for all KoboldAlignX errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize exception with message and optional metadata.
        
        Args:
            message: Human-readable error message
            error_code: Machine-readable error code for logging/monitoring
            details: Additional context about the error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details
        }


# API Client Exceptions
class APIClientError(KoboldAlignXException):
    """Base exception for API client errors."""
    pass


class RelianceAPIError(APIClientError):
    """Error communicating with Reliance API."""
    pass


class KoboldAPIError(APIClientError):
    """Error communicating with Kobold API."""
    pass


class AuthenticationError(APIClientError):
    """Authentication failed with external API."""
    pass


class RateLimitError(APIClientError):
    """API rate limit exceeded."""
    pass


# Data Processing Exceptions
class DataProcessingError(KoboldAlignXException):
    """Base exception for data processing errors."""
    pass


class LocationParsingError(DataProcessingError):
    """Error parsing location data."""
    pass


class DataValidationError(DataProcessingError):
    """Data validation failed."""
    pass


class ExcelGenerationError(DataProcessingError):
    """Error generating Excel file."""
    pass


# Email Exceptions
class EmailError(KoboldAlignXException):
    """Base exception for email-related errors."""
    pass


class EmailSendError(EmailError):
    """Failed to send email."""
    pass


class EmailAttachmentError(EmailError):
    """Error attaching file to email."""
    pass


# Storage Exceptions
class StorageError(KoboldAlignXException):
    """Base exception for storage errors."""
    pass


class JobNotFoundError(StorageError):
    """Requested job not found."""
    pass


class DuplicateJobError(StorageError):
    """Job already exists."""
    pass


# Configuration Exceptions
class ConfigurationError(KoboldAlignXException):
    """Configuration error."""
    pass


class MissingConfigurationError(ConfigurationError):
    """Required configuration value is missing."""
    pass


# Task Exceptions
class TaskError(KoboldAlignXException):
    """Base exception for Celery task errors."""
    pass


class TaskTimeoutError(TaskError):
    """Task execution exceeded time limit."""
    pass


class TaskRetryableError(TaskError):
    """Error that should trigger task retry."""
    pass


# HTTP Exceptions for FastAPI
class HTTPException(KoboldAlignXException):
    """Base HTTP exception with status code."""
    
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize HTTP exception.
        
        Args:
            message: Error message
            status_code: HTTP status code
            error_code: Error code
            details: Additional error details
        """
        super().__init__(message, error_code, details)
        self.status_code = status_code


class BadRequestError(HTTPException):
    """400 Bad Request."""
    
    def __init__(self, message: str = "Bad request", **kwargs):
        super().__init__(message, status_code=400, **kwargs)


class UnauthorizedError(HTTPException):
    """401 Unauthorized."""
    
    def __init__(self, message: str = "Unauthorized", **kwargs):
        super().__init__(message, status_code=401, **kwargs)


class ForbiddenError(HTTPException):
    """403 Forbidden."""
    
    def __init__(self, message: str = "Forbidden", **kwargs):
        super().__init__(message, status_code=403, **kwargs)


class NotFoundError(HTTPException):
    """404 Not Found."""
    
    def __init__(self, message: str = "Resource not found", **kwargs):
        super().__init__(message, status_code=404, **kwargs)


class ConflictError(HTTPException):
    """409 Conflict."""
    
    def __init__(self, message: str = "Resource conflict", **kwargs):
        super().__init__(message, status_code=409, **kwargs)


class InternalServerError(HTTPException):
    """500 Internal Server Error."""
    
    def __init__(self, message: str = "Internal server error", **kwargs):
        super().__init__(message, status_code=500, **kwargs)


class ServiceUnavailableError(HTTPException):
    """503 Service Unavailable."""
    
    def __init__(self, message: str = "Service temporarily unavailable", **kwargs):
        super().__init__(message, status_code=503, **kwargs)
