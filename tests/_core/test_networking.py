import pytest

from axion._core.networking import RateLimitInfo

# Test data for parsing different valid error message formats
RATE_LIMIT_SUCCESS_DATA = [
    (
        'Rate limit exceeded (429): Request too fast. limit=40, remaining=0, reset=12, key=api;minute;40;...',
        40,
        0,
        12,
        'api;minute;40;...',
    ),
    (
        "Too Many Requests. Your request rate is too high. limit=200, remaining=1, reset=5, key='user;hour;200'",
        200,
        1,
        5,
        'user;hour;200',
    ),
    (
        'A 429 error occurred. Details: The maximum call rate was reached. limit=1000\nremaining=0\nreset=30\nkey=global',
        1000,
        0,
        30,
        'global',
    ),
    ('429 limit=5 remaining=3 reset=1 key={client_id};', 5, 3, 1, '{client_id};'),
    (
        'Rate limit reached for gpt-4o in organization org-test on tokens per min (TPM): '
        'Limit 30000, Used 30000, Requested 1809. Please try again in 3.618s.',
        30000,
        0,
        4,
        'gpt-4o;tpm',
    ),
    (
        'Rate limit reached for gpt-4o in organization\\n'
        'org-test on tokens per min (TPM): Limit 30000, Used 27142,\\n'
        'Requested 3440. Please try again in 1.164s.',
        30000,
        2858,
        2,
        'gpt-4o;tpm',
    ),
    (
        'RateLimitError: Anthropic rate limit exceeded. Please try again in 1.2s.',
        0,
        0,
        2,
        'retry-after',
    ),
]

# Test data for messages that should fail parsing
RATE_LIMIT_FAILURE_DATA = [
    'Not a rate limit error (500): Internal server error.',
    'Rate limit but missing data. limit=, remaining=, reset=, key=unknown',
    'Missing components. remaining=10, reset=5',
    '',
    'limit=A remaining=B reset=C key=D',  # Non-integer values
]


@pytest.mark.parametrize(
    'error_message, expected_limit, expected_remaining, expected_reset, expected_key',
    RATE_LIMIT_SUCCESS_DATA,
)
def test_from_error_success(
    error_message, expected_limit, expected_remaining, expected_reset, expected_key
):
    """Tests successful extraction of rate limit parameters from various error messages."""
    info = RateLimitInfo.from_error(error_message)

    # Assert that an object was successfully created
    assert info is not None, 'RateLimitInfo should be created successfully'

    # Assert the extracted values match expectations
    assert info.limit == expected_limit, 'Limit mismatch'
    assert info.remaining == expected_remaining, 'Remaining mismatch'
    assert info.reset_seconds == expected_reset, 'Reset mismatch'


@pytest.mark.parametrize('error_message', RATE_LIMIT_FAILURE_DATA)
def test_from_error_failure(error_message):
    """Tests cases where the error message does not contain parsable rate limit info."""
    info = RateLimitInfo.from_error(error_message)

    # Assert that the parsing failed and returned None
    assert info is None, (
        'RateLimitInfo should be None for non-matching or malformed errors'
    )


def test_get_wait_time_default_buffer():
    """Tests the calculated wait time with the default 1.0 buffer."""
    reset_time = 15
    info = RateLimitInfo(reset_seconds=reset_time)

    expected_wait_time = reset_time + 1.0  # Default buffer is 1.0
    assert info.get_wait_time() == expected_wait_time


def test_get_wait_time_custom_buffer():
    """Tests the calculated wait time with a custom buffer value."""

    reset_time_int = 5
    info = RateLimitInfo(reset_seconds=reset_time_int)
    custom_buffer = 0.5
    expected_wait_time = reset_time_int + custom_buffer  # 5 + 0.5 = 5.5
    assert info.get_wait_time(custom_buffer) == pytest.approx(expected_wait_time)


def test_initialization():
    """Tests direct initialization with custom values."""
    info = RateLimitInfo(limit=10, remaining=5, reset_seconds=8, key='test_key')

    assert info.limit == 10
    assert info.remaining == 5
    assert info.reset_seconds == 8
    assert info.key == 'test_key'


def test_repr():
    """Tests the string representation for debugging."""
    info = RateLimitInfo(limit=50, remaining=1, reset_seconds=10)
    expected_repr = 'RateLimitInfo(limit=50, remaining=1, reset=10s)'

    assert repr(info) == expected_repr
