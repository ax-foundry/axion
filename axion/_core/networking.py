import re
from typing import Optional

from axion._core.logging import get_logger

logger = get_logger(__name__)


class RateLimitInfo:
    """
    Extracts and stores rate limit information from error responses.

    This class parses rate limit errors to extract:
    - limit: Total requests allowed per time window
    - remaining: Requests remaining in current window
    - reset_seconds: Time until rate limit resets
    - key: Rate limit identifier
    """

    def __init__(
        self, limit: int = 0, remaining: int = 0, reset_seconds: int = 0, key: str = ''
    ):
        self.limit = limit
        self.remaining = remaining
        self.reset_seconds = reset_seconds
        self.key = key

    @classmethod
    def from_error(cls, error_message: str) -> Optional['RateLimitInfo']:
        """
        Parse rate limit information from error message.

        Args:
            error_message: The error message string to parse

        Returns:
            RateLimitInfo object if parsing succeeds, None otherwise
        """
        try:
            # First check if this is a 500-wrapped 429 error and extract the inner message
            wrapped_pattern = r"Error code: 500.*?'detail':\s*'500:\s*429:\s*({.*?})'"
            wrapped_match = re.search(wrapped_pattern, error_message, re.DOTALL)

            if wrapped_match:
                # Extract the inner JSON-like message
                inner_message = wrapped_match.group(1)
                logger.debug(
                    f'Detected wrapped 500 error, extracting inner message: {inner_message[:100]}...'
                )
                error_message = inner_message

            # Look for the pattern: {limit=X, remaining=Y, reset=Z, key=...}
            # We use a non-greedy match (.*?) for the intermediate fields.
            pattern = r'limit[=:](\d+).*?remaining[=:](\d+).*?reset[=:](\d+).*?key[=:]([^,}]+)'
            match = re.search(pattern, error_message, re.DOTALL)

            if match:
                limit = int(match.group(1))
                remaining = int(match.group(2))
                reset_seconds = int(match.group(3))

                # Apply strip() first to remove any potential leading/trailing whitespace
                # or newlines captured by the regex, then strip quotes.
                key = match.group(4).strip().strip('"\'')

                logger.info(
                    f'📊 Rate limit detected: {remaining}/{limit} remaining, '
                    f'resets in {reset_seconds}s (key: {key})'
                )

                return cls(
                    limit=limit,
                    remaining=remaining,
                    reset_seconds=reset_seconds,
                    key=key,
                )
        except Exception as e:
            logger.debug(f'Could not parse rate limit info: {e}')

        return None

    def get_wait_time(self, buffer_seconds: float = 1.0) -> float:
        """
        Calculate how long to wait based on rate limit reset time.

        Args:
            buffer_seconds: Additional buffer time to add to reset_seconds

        Returns:
            Number of seconds to wait
        """
        # Add a buffer to account for clock skew and ensure we're past the reset
        return self.reset_seconds + buffer_seconds

    def __repr__(self) -> str:
        return (
            f'RateLimitInfo(limit={self.limit}, remaining={self.remaining}, '
            f'reset={self.reset_seconds}s)'
        )
