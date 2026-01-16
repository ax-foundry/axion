import os
from typing import Any, Dict, Literal, Optional, Union

import requests

from axion._core.logging import get_logger
from axion._core.metadata.schema import ToolMetadata
from axion._core.tracing import init_tracer, trace
from axion._core.tracing.handlers import BaseTraceHandler
from axion._core.utils import Timer, log_execution_time

logger = get_logger(__name__)


class BaseAPI:
    """
    Base API class with unified tracing capabilities.
    Provides OAuth authentication and request handling for API integrations.
    """

    _metadata_type = 'base'

    def __init__(
        self,
        domain: Optional[str] = None,
        consumer_key: Optional[str] = None,
        consumer_secret: Optional[str] = None,
        token: Optional[str] = None,
        auth_type: Optional[
            Literal['password', 'client_credentials']
        ] = 'client_credentials',
        tracer: Optional[BaseTraceHandler] = None,
        **kwargs,
    ):
        self._session_id = None
        self.domain = domain or os.environ.get('API_DOMAIN')
        self._consumer_key = consumer_key or os.environ.get('API_CONSUMER_KEY')
        self._consumer_secret = consumer_secret or os.environ.get('API_CONSUMER_SECRET')
        self._token = token
        self._auth_type = auth_type
        self._username = kwargs.get('username', None)
        self._password = kwargs.get('password', None)

        self._init_tracer(tracer)

        if not self._token:
            self._token = self.get_oauth_token()['access_token']
        logger.info(f'{self.__class__.__name__} initialized for domain: {self.domain}')

    def _init_tracer(self, tracer: Optional[BaseTraceHandler] = None):
        """Initialize tracer with appropriate metadata for API operations."""
        tool_metadata = ToolMetadata(
            name=f'{self.__class__.__name__.lower()}',
            description=f'{self.__class__.__name__} API client',
            owner='Axion',
            version='1.0.0',
        )

        self.tracer = init_tracer(self._metadata_type, tool_metadata, tracer)

    @trace(name='get_oauth_token', capture_args=False)
    def get_oauth_token(self, refresh: Optional[bool] = False) -> dict:
        """
        Retrieve an OAuth2 token via client credentials grant.

        Returns:
            dict: The JSON response containing the access token.

        Raises:
            HTTPError: If the HTTP request returned an unsuccessful status code.
        """
        # Use cached token
        if self._token and not refresh:
            logger.info('Using cached OAuth token.')
            if self.tracer:
                with self.tracer.span(
                    'token_cache_hit', auth_type=self._auth_type
                ) as span:
                    span.set_attribute('cached_token_used', True)
            return {'access_token': self._token}

        # Refresh token
        if self.tracer:
            with self.tracer.span(
                'oauth_request',
                domain=self.domain,
                auth_type=self._auth_type,
                refresh=refresh,
            ) as span:
                return self._perform_oauth_request(span)
        else:
            return self._perform_oauth_request(None)

    def _perform_oauth_request(self, span=None) -> dict:
        """Perform the actual OAuth request with optional span tracking."""
        url = f'https://{self.domain}/services/oauth2/token'
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}

        match self._auth_type:
            case 'client_credentials':
                payload = {
                    'grant_type': 'client_credentials',
                    'client_id': self._consumer_key,
                    'client_secret': self._consumer_secret,
                }
            case 'password':
                payload = {
                    'grant_type': 'password',
                    'client_id': self._consumer_key,
                    'client_secret': self._consumer_secret,
                    'username': self._username,
                    'password': self._password,
                }
            case _:
                raise ValueError(
                    "Invalid auth type. Valid values: 'password' or 'client_credentials'."
                )

        if span:
            span.set_attribute('oauth_url', url)
            span.set_attribute('grant_type', payload['grant_type'])

        logger.info('Fetching a new OAuth token.')

        try:
            with Timer() as timer:
                response = requests.post(url, headers=headers, data=payload)
                response.raise_for_status()  # Raise an exception for HTTP errors

            self._token = response.json()['access_token']

            if span:
                span.set_attribute('oauth_success', True)
                elapsed_time = (
                    round(timer.elapsed_time * 1000, 2) if timer.elapsed_time else 0
                )
                span.set_attribute('oauth_latency', elapsed_time)
                span.set_attribute('response_status', response.status_code)

            return response.json()

        except Exception as e:
            if span:
                span.set_attribute('oauth_success', False)
                span.set_attribute('error_type', type(e).__name__)
                span.set_attribute('error_message', str(e)[:200])
            raise

    @log_execution_time
    def _make_request(
        self,
        method: str,
        url: str,
        headers: dict,
        payload: dict,
        retry: Optional[bool] = True,
    ) -> dict:
        """
        Make a request to the external API with tracing.

        Args:
            method (str): The HTTP method to use.
            url (str): The URL to send the request to.
            headers (dict): The request headers.
            payload (dict): The request payload.
            retry (bool, optional): Whether to retry the request if the token has expired. Defaults to True.

        Returns:
            dict: The JSON response containing the request result.

        Raises:
            HTTPError: If the HTTP request returned an unsuccessful status code.
        """
        if self.tracer:
            with self.tracer.span(
                'api_request',
                method=method,
                url_domain=self.domain,
                retry_enabled=retry,
                payload_size=len(str(payload)) if payload else 0,
            ) as span:
                return self._perform_api_request(
                    method, url, headers, payload, retry, span
                )
        else:
            return self._perform_api_request(method, url, headers, payload, retry, None)

    def _perform_api_request(
        self,
        method: str,
        url: str,
        headers: dict,
        payload: dict,
        retry: bool,
        span=None,
    ) -> dict:
        """Perform the actual API request with optional span tracking."""
        default_headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self._token}',
        }
        final_headers = {**default_headers, **headers}

        if span:
            span.set_attribute('request_url', url)
            span.set_attribute('has_custom_headers', bool(headers))

        try:
            with Timer() as timer:
                response = requests.request(
                    method, url, headers=final_headers, json=payload
                )

            if span:
                span.set_attribute('response_status', response.status_code)
                elapsed_time = (
                    round(timer.elapsed_time * 1000, 2) if timer.elapsed_time else 0
                )
                span.set_attribute('request_latency', elapsed_time)
                span.set_attribute(
                    'response_size', len(response.text) if response.text else 0
                )

            # Handle token expiration
            if response.status_code == 401 and retry:
                if span:
                    span.set_attribute('token_expired', True)
                    span.set_attribute('retrying_request', True)

                logger.error(
                    f'Failed with status code {response.status_code}. error message: {response.text}'
                )
                logger.error('Token expired. Refreshing token and retrying request.')
                self.get_oauth_token(refresh=True)
                return self._make_request(method, url, headers, payload, retry=False)

            response.raise_for_status()
            result = response.json()

            if span:
                span.set_attribute('request_success', True)
                if isinstance(result, dict):
                    span.set_attribute(
                        'response_keys', list(result.keys())[:10]
                    )  # Limit to 10 keys

            return result

        except Exception as e:
            if span:
                span.set_attribute('request_success', False)
                span.set_attribute('error_type', type(e).__name__)
                span.set_attribute('error_message', str(e)[:200])
            raise

    @property
    def execution_time(self) -> Union[float, None]:
        """
        Execution time of the last request.
        """
        return getattr(self, '_execution_time', None)

    def display_api_statistics(self):
        """Display API usage statistics."""
        if self.tracer:
            if hasattr(self.tracer, 'display_api_statistics'):
                self.tracer.display_api_statistics()
            else:
                self.tracer.display_traces()
        else:
            logger.info('Tracing not enabled for this API instance')

    def get_execution_metadata(self) -> Dict[str, Any]:
        """Get comprehensive execution metadata."""
        base_metadata = {
            'api_class': self.__class__.__name__,
            'domain': self.domain,
            'auth_type': self._auth_type,
            'has_session': bool(self._session_id),
        }

        if self.tracer:
            return {**base_metadata, **self.tracer.get_metadata()}

        return base_metadata
