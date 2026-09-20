class MetricValidationError(Exception):
    """Raised when metric validation fails."""

    pass


class MetricRegistryError(Exception):
    """Raised when there's an error with the metric registry operations."""

    pass


class JevError(Exception):
    """Raised when a call to the Jev decision endpoint does not produce answers."""

    pass


class JevAuthError(JevError):
    """Raised when Jev rejects the credential (401/403)."""

    pass


class JevRateLimitError(JevError):
    """Raised when Jev rate-limits the call (429)."""

    pass
