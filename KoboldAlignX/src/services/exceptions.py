"""
Custom exceptions for API client operations.

This module provides a hierarchy of exceptions for handling various API-related errors:
- APIClientError: Base exception for all API client errors
- AuthenticationError: OAuth/authentication failures
- APIRequestError: Network/HTTP request failures
- ValidationError: Response validation failures


"""

from typing import Optional, Dict, Any, List
import pandas as pd


class APIClientError(Exception):
    """
    Base exception for all API client errors.
    
    All custom exceptions in this module inherit from this class,
    allowing you to catch all API-related errors with a single except clause.
    
    """
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ):
        self.message = message
        self.details = details or {}
        self.original_exception = original_exception
        
        # Call parent constructor
        super().__init__(self.message)
    
    def __str__(self) -> str:
        """Return string representation of the error."""
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message
    
    def __repr__(self) -> str:
        """Return detailed representation of the error."""
        return f"{self.__class__.__name__}(message={self.message!r}, details={self.details!r})"
    
    def to_dict(self) -> Dict[str, Any]:
        """ Convert exception to dictionary for logging or serialization. """
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'details': self.details,
            'original_exception': str(self.original_exception) if self.original_exception else None
        }


class AuthenticationError(APIClientError):
    """ Raised when authentication fails."""
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ):
        super().__init__(message, details, original_exception)


class APIRequestError(APIClientError):
    """ Raised when an API request fails."""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ):
        super().__init__(message, details, original_exception)
    
    @property
    def is_retriable(self) -> bool:
        # Server errors (5xx) are typically retriable
        status_code = self.details.get('status_code')
        if status_code and 500 <= status_code < 600:
            return True
        
        # Timeout errors are retriable
        if 'timeout' in self.message.lower():
            return True
        
        # Connection errors are retriable
        if 'connection' in self.message.lower():
            return True
        
        return False


class ValidationError(APIClientError):
    """ Raised when response validation fails."""
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ):
        super().__init__(message, details, original_exception)
    
    @property
    def missing_fields(self) -> list:
        """
        Get list of missing required fields.
        
        Returns:
            List of field names that were missing
        """
        return self.details.get('missing_fields', [])

class DataProcessingError(Exception):
    """
    Base exception for all data processing errors.
    
    All custom exceptions in this module inherit from this class,
    allowing you to catch all data processing errors with a single except clause.
    
    """
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
        data_info: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.details = details or {}
        self.original_exception = original_exception
        self.data_info = data_info or {}
        
        # Call parent constructor
        super().__init__(self.message)
    
    def __str__(self) -> str:
        """Return string representation of the error."""
        parts = [self.message]
        
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            parts.append(f"({details_str})")
        
        if self.data_info:
            data_str = ", ".join(f"{k}={v}" for k, v in self.data_info.items())
            parts.append(f"[{data_str}]")
        
        return " ".join(parts)
    
    def __repr__(self) -> str:
        """Return detailed representation of the error."""
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"details={self.details!r}, "
            f"data_info={self.data_info!r})"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'details': self.details,
            'data_info': self.data_info,
            'original_exception': str(self.original_exception) if self.original_exception else None
        }
    
    def add_context(self, **kwargs) -> 'DataProcessingError':        
        self.details.update(kwargs)
        return self


# ============================================================================
# Specific Exceptions
# ============================================================================

class DataValidationError(DataProcessingError):
    """
    Raised when data validation fails.
    
    This includes:
    - Missing required columns
    - Invalid data types
    - Missing required fields
    - Invalid data ranges
    - Schema mismatches
    
    """
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
        data_info: Optional[Dict[str, Any]] = None,
        missing_columns: Optional[List[str]] = None,
        invalid_columns: Optional[List[str]] = None
    ):
        super().__init__(message, details, original_exception, data_info)
        self._missing_columns = missing_columns or []
        self._invalid_columns = invalid_columns or []
        
        # Add to details if provided
        if missing_columns:
            self.details['missing_columns'] = missing_columns
        if invalid_columns:
            self.details['invalid_columns'] = invalid_columns
    
    @property
    def missing_columns(self) -> List[str]:
        """Get list of missing columns."""
        return self._missing_columns or self.details.get('missing_columns', [])
    
    @property
    def invalid_columns(self) -> List[str]:
        """Get list of invalid columns."""
        return self._invalid_columns or self.details.get('invalid_columns', [])


class DataAlignmentError(DataProcessingError):
    """
    Raised when data alignment fails.
    
    This includes:
    - Cross-correlation failures
    - Incompatible dataset lengths
    - Missing reference data
    - Peak detection failures
    
    """
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
        data_info: Optional[Dict[str, Any]] = None,
        datasets: Optional[List[str]] = None
    ):
        super().__init__(message, details, original_exception, data_info)
        self._datasets = datasets or []
        
        if datasets:
            self.details['datasets'] = datasets
    
    @property
    def datasets(self) -> List[str]:
        """Get list of datasets involved in alignment failure."""
        return self._datasets or self.details.get('datasets', [])


class EmptyDataError(DataProcessingError):
    """
    Raised when required data is empty or missing.
    
    This includes:
    - Empty dataframes
    - No records returned from API
    - Missing required datasets
    - Zero-length arrays
    
    """
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
        data_info: Optional[Dict[str, Any]] = None,
        dataset_name: Optional[str] = None
    ):
        super().__init__(message, details, original_exception, data_info)
        self._dataset_name = dataset_name
        
        if dataset_name:
            self.details['dataset_name'] = dataset_name
    
    @property
    def dataset_name(self) -> Optional[str]:
        """Get the name of the empty dataset."""
        return self._dataset_name or self.details.get('dataset_name')


class TimingError(DataProcessingError):
    """
    Raised when timing correction fails.
    
    This includes:
    - Unable to fix timing irregularities
    - Invalid time deltas
    - Resampling failures
    - Timestamp conflicts
    """
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
        data_info: Optional[Dict[str, Any]] = None,
        unfixable_sections: Optional[int] = None
    ):
        super().__init__(message, details, original_exception, data_info)
        self._unfixable_sections = unfixable_sections
        
        if unfixable_sections is not None:
            self.details['unfixable_sections'] = unfixable_sections
    
    @property
    def unfixable_sections(self) -> Optional[int]:
        """Get the number of unfixable timing sections."""
        return self._unfixable_sections or self.details.get('unfixable_sections')


# ============================================================================
# Helper Functions
# ============================================================================

def create_validation_error(
    message: str,
    missing_columns: Optional[List[str]] = None,
    invalid_columns: Optional[List[str]] = None,
    df: Optional[pd.DataFrame] = None,
    original: Optional[Exception] = None
) -> DataValidationError:
    data_info = {}
    if df is not None:
        data_info = {
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': list(df.columns)
        }
    
    return DataValidationError(
        message,
        data_info=data_info,
        missing_columns=missing_columns,
        invalid_columns=invalid_columns,
        original_exception=original
    )


def create_alignment_error(
    message: str,
    datasets: Optional[List[str]] = None,
    shift: Optional[int] = None,
    correlation: Optional[float] = None,
    original: Optional[Exception] = None
) -> DataAlignmentError:
    details = {}
    if shift is not None:
        details['shift'] = shift
    if correlation is not None:
        details['correlation'] = correlation
    
    return DataAlignmentError(
        message,
        details=details,
        datasets=datasets,
        original_exception=original
    )


def create_empty_data_error(
    message: str,
    dataset_name: Optional[str] = None,
    expected_rows: Optional[int] = None,
    original: Optional[Exception] = None
) -> EmptyDataError:
    details = {}
    if expected_rows is not None:
        details['expected_rows'] = expected_rows
    
    return EmptyDataError(
        message,
        details=details,
        dataset_name=dataset_name,
        original_exception=original
    )


def create_timing_error(
    message: str,
    unfixable_sections: Optional[int] = None,
    total_sections: Optional[int] = None,
    original: Optional[Exception] = None
) -> TimingError:
    details = {}
    if unfixable_sections is not None:
        details['unfixable_sections'] = unfixable_sections
    if total_sections is not None:
        details['total_sections'] = total_sections
    
    return TimingError(
        message,
        details=details,
        unfixable_sections=unfixable_sections,
        original_exception=original
    )


# ============================================================================
# Type Guards
# ============================================================================

def is_validation_error(error: Exception) -> bool:
    """Check if error is a DataValidationError."""
    return isinstance(error, DataValidationError)


def is_alignment_error(error: Exception) -> bool:
    """Check if error is a DataAlignmentError."""
    return isinstance(error, DataAlignmentError)


def is_empty_data_error(error: Exception) -> bool:
    """Check if error is an EmptyDataError."""
    return isinstance(error, EmptyDataError)


def is_timing_error(error: Exception) -> bool:
    """Check if error is a TimingError."""
    return isinstance(error, TimingError)


def is_data_processing_error(error: Exception) -> bool:
    """Check if error is any DataProcessingError."""
    return isinstance(error, DataProcessingError)


# ============================================================================
# Validation Helpers
# ============================================================================

def validate_dataframe_structure(
    df: pd.DataFrame,
    required_columns: List[str],
    name: str = "DataFrame"
) -> None:
    if df is None:
        raise create_empty_data_error(
            f"{name} is None",
            dataset_name=name
        )
    
    if df.empty:
        raise create_empty_data_error(
            f"{name} is empty",
            dataset_name=name
        )
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise create_validation_error(
            f"{name} missing required columns",
            missing_columns=missing_columns,
            df=df
        )


def validate_data_types(
    df: pd.DataFrame,
    column_types: Dict[str, type],
    name: str = "DataFrame"
) -> None:
    invalid_columns = []
    
    for col, expected_type in column_types.items():
        if col not in df.columns:
            continue
            
        # Check if column has the expected type
        if expected_type == float or expected_type == int:
            if not pd.api.types.is_numeric_dtype(df[col]):
                invalid_columns.append(col)
        elif expected_type == pd.Timestamp:
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                invalid_columns.append(col)
    
    if invalid_columns:
        raise create_validation_error(
            f"{name} has columns with invalid types",
            invalid_columns=invalid_columns,
            df=df
        )


def validate_data_range(
    df: pd.DataFrame,
    column: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    name: str = "DataFrame"
) -> None:    
    if column not in df.columns:
        raise create_validation_error(
            f"{name} missing column for range validation",
            missing_columns=[column],
            df=df
        )
    
    if min_value is not None:
        below_min = (df[column] < min_value).sum()
        if below_min > 0:
            raise DataValidationError(
                f"{name} has {below_min} values below minimum",
                details={
                    'column': column,
                    'min_value': min_value,
                    'violations': below_min
                }
            )
    
    if max_value is not None:
        above_max = (df[column] > max_value).sum()
        if above_max > 0:
            raise DataValidationError(
                f"{name} has {above_max} values above maximum",
                details={
                    'column': column,
                    'max_value': max_value,
                    'violations': above_max
                }
            )
        
# Helper functions for creating exceptions with context

def create_auth_error(
    message: str,
    status_code: Optional[int] = None,
    endpoint: Optional[str] = None,
    original: Optional[Exception] = None
) -> AuthenticationError:
    """ Create an AuthenticationError with standard details."""
    details = {}
    if status_code:
        details['status_code'] = status_code
    if endpoint:
        details['endpoint'] = endpoint
    
    return AuthenticationError(message, details, original)


def create_request_error(
    message: str,
    url: Optional[str] = None,
    status_code: Optional[int] = None,
    attempt: Optional[int] = None,
    max_attempts: Optional[int] = None,
    original: Optional[Exception] = None
) -> APIRequestError:
    """ Create an APIRequestError with standard details. """
    details = {}
    if url:
        details['url'] = url
    if status_code:
        details['status_code'] = status_code
    if attempt:
        details['attempt'] = attempt
    if max_attempts:
        details['max_attempts'] = max_attempts
    
    return APIRequestError(message, details, original)


def create_validation_error(
    message: str,
    missing_fields: Optional[list] = None,
    invalid_type: Optional[str] = None,
    expected_type: Optional[str] = None,
    actual_type: Optional[str] = None,
    original: Optional[Exception] = None
) -> ValidationError:
    """ Create a ValidationError with standard details. """
    details = {}
    if missing_fields:
        details['missing_fields'] = missing_fields
    if invalid_type:
        details['invalid_type'] = invalid_type
    if expected_type:
        details['expected_type'] = expected_type
    if actual_type:
        details['actual_type'] = actual_type
    
    return ValidationError(message, details, original)


# Type guards for isinstance checks

def is_auth_error(error: Exception) -> bool:
    """Check if error is an AuthenticationError."""
    return isinstance(error, AuthenticationError)


def is_request_error(error: Exception) -> bool:
    """Check if error is an APIRequestError."""
    return isinstance(error, APIRequestError)


def is_validation_error(error: Exception) -> bool:
    """Check if error is a ValidationError."""
    return isinstance(error, ValidationError)


def is_api_error(error: Exception) -> bool:
    """Check if error is any APIClientError."""
    return isinstance(error, APIClientError)