"""API client classes for interacting with external services.

This module provides robust API clients with:
- Automatic retry logic with exponential backoff
- Comprehensive error handling
- Request/response validation
- Detailed logging
- Timeout management
"""
import logging
from base64 import b64encode
from typing import Optional, Dict, List, Set, Any
from dataclasses import dataclass
import asyncio
import aiohttp
import requests

from config import (
    API_BASE_URL,
    RELIANCE_OAUTH_URL,
    RELIANCE_JOBS_URL,
    OAUTH_CLIENT_ID,
    OAUTH_CLIENT_SECRET
)

from exceptions import (
    AuthenticationError,
    APIRequestError,
    ValidationError
)

logger = logging.getLogger(__name__)


# Custom Exceptions
class APIClientError(Exception):
    """Base exception for API client errors."""
    pass


class AuthenticationError(APIClientError):
    """Raised when authentication fails."""
    pass


class APIRequestError(APIClientError):
    """Raised when an API request fails."""
    pass


class ValidationError(APIClientError):
    """Raised when response validation fails."""
    pass


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt number."""
        delay = self.base_delay * (self.exponential_base ** attempt)
        return min(delay, self.max_delay)


@dataclass
class TimeoutConfig:
    """Configuration for request timeouts."""
    total: int = 300  # 5 minutes total
    connect: int = 10  # 10 seconds to connect
    sock_read: int = 60  # 1 minute to read response


class BaseAPIClient:
    """Base class for API clients with common functionality."""
    
    def __init__(
        self,
        session: aiohttp.ClientSession,
        retry_config: Optional[RetryConfig] = None,
        timeout_config: Optional[TimeoutConfig] = None
    ):
        self.session = session
        self.retry_config = retry_config or RetryConfig()
        self.timeout_config = timeout_config or TimeoutConfig()
        self._timeout = aiohttp.ClientTimeout(
            total=self.timeout_config.total,
            connect=self.timeout_config.connect,
            sock_read=self.timeout_config.sock_read
        )
    
    async def _make_request(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> aiohttp.ClientResponse:       
        kwargs.setdefault('timeout', self._timeout)
        
        for attempt in range(self.retry_config.max_attempts):
            try:
                logger.debug(
                    f"Making {method} request to {url} (attempt {attempt + 1}/{self.retry_config.max_attempts})"
                )
                
                async with self.session.request(method, url, **kwargs) as response:
                    # Log response details
                    logger.debug(f"Response status: {response.status} for {url}")
                    
                    # Return response for successful requests
                    if response.status < 500:
                        return response
                    
                    # Server error - might be temporary, retry
                    logger.warning(
                        f"Server error {response.status} for {url}, "
                        f"attempt {attempt + 1}/{self.retry_config.max_attempts}"
                    )
                    
            except asyncio.TimeoutError as e:
                logger.warning(f"Request timeout for {url}, attempt {attempt + 1}")
                if attempt == self.retry_config.max_attempts - 1:
                    raise APIRequestError(f"Request timeout after {self.retry_config.max_attempts} attempts") from e
                    
            except aiohttp.ClientError as e:
                logger.warning(f"Client error for {url}: {str(e)}, attempt {attempt + 1}")
                if attempt == self.retry_config.max_attempts - 1:
                    raise APIRequestError(f"Request failed after {self.retry_config.max_attempts} attempts: {str(e)}") from e
            
            # Wait before retrying (except on last attempt)
            if attempt < self.retry_config.max_attempts - 1:
                delay = self.retry_config.get_delay(attempt)
                logger.debug(f"Waiting {delay:.2f}s before retry")
                await asyncio.sleep(delay)
        
        raise APIRequestError(f"Request failed after {self.retry_config.max_attempts} attempts")
    
    async def _get_json(self, url: str, **kwargs) -> Dict[str, Any]:
        """ Make GET request and return JSON response. """
        response = await self._make_request('GET', url, **kwargs)
        
        try:
            data = await response.json()
            return data
        except (aiohttp.ContentTypeError, ValueError) as e:
            logger.error(f"Failed to parse JSON response from {url}")
            raise ValidationError(f"Invalid JSON response: {str(e)}") from e
    
    async def _post_json(self, url: str, json: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """ Make POST request with JSON payload and return JSON response. """
        response = await self._make_request('POST', url, json=json, **kwargs)
        
        try:
            data = await response.json()
            return data
        except (aiohttp.ContentTypeError, ValueError) as e:
            logger.error(f"Failed to parse JSON response from {url}")
            raise ValidationError(f"Invalid JSON response: {str(e)}") from e
    
    def _validate_response(self, data: Dict[str, Any], required_fields: List[str]) -> None:
        """ Validate response contains required fields. """
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValidationError(f"Missing required fields in response: {missing_fields}")


class RelianceAPIClient(BaseAPIClient):
    """Client for interacting with the Reliance API."""
    
    def __init__(
        self,
        session: aiohttp.ClientSession,
        retry_config: Optional[RetryConfig] = None,
        timeout_config: Optional[TimeoutConfig] = None
    ):
        super().__init__(session, retry_config, timeout_config)
        self.token: Optional[str] = None
    
    async def get_oauth_token(self) -> str:
        """ Obtain OAuth token for API authentication. """
        credentials = f"{OAUTH_CLIENT_ID}:{OAUTH_CLIENT_SECRET}"
        encoded_credentials = b64encode(credentials.encode()).decode()
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Basic {encoded_credentials}",
        }
        payload = {"grant_type": "client_credentials"}
        
        try:
            logger.info("Requesting OAuth token")
            data = await self._post_json(
                RELIANCE_OAUTH_URL,
                json=payload,
                headers=headers
            )
            
            # Validate response
            self._validate_response(data, ['access_token'])
            
            self.token = data["access_token"]
            logger.info("Successfully obtained OAuth token")
            return self.token
            
        except APIRequestError as e:
            logger.error(f"Failed to obtain OAuth token: {str(e)}")
            raise AuthenticationError(f"Authentication failed: {str(e)}") from e
        except ValidationError as e:
            logger.error(f"Invalid OAuth response: {str(e)}")
            raise AuthenticationError(f"Invalid authentication response: {str(e)}") from e
    
    def _ensure_authenticated(self) -> None:
        """ Ensure client is authenticated. """
        if not self.token:
            raise AuthenticationError("No OAuth token available. Call get_oauth_token() first.")
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get headers with authentication token."""
        self._ensure_authenticated()
        return {"Authorization": f"Bearer {self.token}"}
    
    async def get_jobs(self) -> List[Dict[str, Any]]:
        """ Fetch all jobs from the Reliance API. """
        try:
            logger.info("Fetching jobs from Reliance API")
            
            data = await self._get_json(
                RELIANCE_JOBS_URL,
                headers=self._get_auth_headers()
            )
            
            # Validate response structure
            self._validate_response(data, ['result'])
            
            jobs = data["result"]
            logger.info(f"Successfully fetched {len(jobs)} jobs")
            return jobs
            
        except AuthenticationError:
            raise
        except (APIRequestError, ValidationError) as e:
            logger.error(f"Failed to fetch jobs: {str(e)}")
            raise
    
    async def get_job_info(self, job_id: int) -> List[str]:
        """ Fetch job header information."""
        try:
            logger.info(f"Fetching job info for job_id={job_id}")
            
            url = f"{RELIANCE_JOBS_URL}/{job_id}/info"
            data = await self._get_json(url, headers=self._get_auth_headers())
            
            # Validate response structure
            self._validate_response(data, ['result'])
            
            if not data['result'] or not data['result'][0]:
                raise ValidationError(f"Empty result for job_id={job_id}")
            
            if 'header' not in data['result'][0]:
                raise ValidationError(f"Missing 'header' field for job_id={job_id}")
            
            job_header = data["result"][0]["header"]
            logger.info(f"Successfully fetched {len(job_header)} columns for job_id={job_id}")
            return job_header
            
        except AuthenticationError:
            raise
        except (APIRequestError, ValidationError) as e:
            logger.error(f"Failed to fetch job info for job_id={job_id}: {str(e)}")
            raise
    
    async def get_job_data(self, job_id: int) -> Dict[str, Any]:
        """ Fetch job data from Reliance API. """
        try:
            logger.info(f"Fetching job data for job_id={job_id}")
            
            url = f"{RELIANCE_JOBS_URL}/{job_id}/data"
            data = await self._get_json(url, headers=self._get_auth_headers())
            
            # Validate response structure
            self._validate_response(data, ['result'])
            
            logger.info(f"Successfully fetched job data for job_id={job_id}")
            return data
            
        except AuthenticationError:
            raise
        except (APIRequestError, ValidationError) as e:
            logger.error(f"Failed to fetch job data for job_id={job_id}: {str(e)}")
            raise


class InternalAPIClient(BaseAPIClient):
    """Client for interacting with the internal Kobold API."""
    
    async def get_processed_jobs(self) -> Set[str]:
        try:
            logger.info("Fetching processed jobs")
            
            url = f"{API_BASE_URL}/processedjobs"
            data = await self._get_json(url)
            
            # Validate that we got a list
            if not isinstance(data, list):
                raise ValidationError(f"Expected list, got {type(data).__name__}")
            
            # Extract job IDs
            reliance_job_ids = set()
            for job in data:
                if not isinstance(job, dict):
                    logger.warning(f"Skipping invalid job entry: {job}")
                    continue
                
                if 'reliance_job_id' not in job:
                    logger.warning(f"Job missing reliance_job_id: {job}")
                    continue
                
                reliance_job_ids.add(str(job['reliance_job_id']))
            
            logger.info(f"Successfully fetched {len(reliance_job_ids)} processed jobs")
            return reliance_job_ids
            
        except (APIRequestError, ValidationError) as e:
            logger.error(f"Failed to fetch processed jobs: {str(e)}")
            raise
    
    def save_processed_job(
        self,
        kobold_job_id: str,
        reliance_job_id: str,
        uwi: str
    ) -> bool:
        url = f"{API_BASE_URL}/processedjobs"
        headers = {"Content-Type": "application/json"}
        payload = {
            'kobold_job_id': str(kobold_job_id),
            'reliance_job_id': str(reliance_job_id),
            'uwi': str(uwi)
        }
        
        try:
            logger.info(f"Saving processed job: reliance_job_id={reliance_job_id}")
            
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 201:
                logger.info(f"Successfully saved processed job {reliance_job_id}")
                return True
            else:
                logger.error(
                    f"Failed to save processed job {reliance_job_id}: "
                    f"status={response.status_code}, response={response.text}"
                )
                return False
                
        except requests.Timeout:
            logger.error(f"Timeout saving processed job {reliance_job_id}")
            return False
        except requests.RequestException as e:
            logger.error(f"Request error saving processed job {reliance_job_id}: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error saving processed job {reliance_job_id}: {str(e)}")
            return False
    
    async def get_guidehawk_data(
        self,
        uwi: str,
        job_type: str = 'RO'
    ) -> Dict[str, Any]:
        try:
            logger.info(f"Fetching GuideHawk data for uwi={uwi}, job_type={job_type}")
            
            url = f"{API_BASE_URL}/guidehawkdatauwi/{uwi}"
            params = {'jobtype': job_type}
            
            # Add params to URL
            url_with_params = f"{url}?jobtype={job_type}"
            
            data = await self._get_json(url_with_params)
            
            # Validate that we got a list (GuideHawk returns array of records)
            if not isinstance(data, list):
                raise ValidationError(f"Expected list, got {type(data).__name__}")
            
            logger.info(f"Successfully fetched {len(data)} GuideHawk records for uwi={uwi}")
            return data
            
        except (APIRequestError, ValidationError) as e:
            logger.error(f"Failed to fetch GuideHawk data for uwi={uwi}: {str(e)}")
            raise


# Factory functions for easy client creation
def create_reliance_client(
    session: aiohttp.ClientSession,
    max_retries: int = 3,
    timeout_seconds: int = 300
) -> RelianceAPIClient:
    retry_config = RetryConfig(max_attempts=max_retries)
    timeout_config = TimeoutConfig(total=timeout_seconds)
    return RelianceAPIClient(session, retry_config, timeout_config)


def create_internal_client(
    session: aiohttp.ClientSession,
    max_retries: int = 3,
    timeout_seconds: int = 300
) -> InternalAPIClient:
    retry_config = RetryConfig(max_attempts=max_retries)
    timeout_config = TimeoutConfig(total=timeout_seconds)
    return InternalAPIClient(session, retry_config, timeout_config)