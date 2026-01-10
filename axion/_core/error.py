from __future__ import annotations


class InvalidRun(Exception):
    pass


class InvalidConfig(Exception):
    """Raised when the prompt configuration is invalid."""

    pass


class GenerationError(Exception):
    """Base exception for generation errors."""

    pass


class ValidationError(GenerationError):
    """Raised when model validation fails."""

    pass


class CustomBaseException(Exception):
    """
    Base exception class.
    """

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class CustomValidationError(Exception):
    """Custom exception that wraps Pydantic ValidationError with better formatting."""

    def __init__(self, message: str, original_error: ValidationError = None):
        self.message = message
        self.original_error = original_error
        super().__init__(message)


class DatabaseError(Exception):
    """Base exception for database operations."""

    pass


class QueryError(DatabaseError):
    """Exception raised when query execution fails."""

    pass


class DataProcessingError(DatabaseError):
    """Exception raised when data processing fails."""

    pass


class ExceptionInRunner(CustomBaseException):
    """
    Exception raised when an exception is raised in the executor.
    """

    def __init__(self):
        msg = (
            'The runner thread which was running the jobs raised an exception. '
            'Read the traceback above to debug it. You can also pass `raise_exceptions=False` '
            'incase you want to show only a warning message instead.'
        )
        super().__init__(msg)


class AIOutputParserException(CustomBaseException):
    """
    Exception raised when the output parser fails to parse the output.
    """

    def __init__(self):
        msg = 'The output parser failed to parse the output including retries.'
        super().__init__(msg)


class LocalPlatformError(Exception):
    """
    Raised when capability is not available on local platform.
    """

    def __init__(self, message='Capability is not available on Local platform'):
        self.message = message
        super().__init__(self.message)


class ParsingError(GenerationError):
    """Raised when output parsing fails."""

    pass
