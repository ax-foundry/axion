import asyncio
import logging

import pytest
from axion._core.logging import (
    DEFAULT_LOG_LEVEL,
    RICH_AVAILABLE,
    _log_stats,
    _to_bool,
    configure_logging,
    get_logger,
)


@pytest.fixture(autouse=True)
def reset_logging_state(monkeypatch):
    """Resets the global state of the logging module before each test."""
    # Clear any logged stats
    _log_stats.clear()
    monkeypatch.setattr('axion._core.logging._logging_configured', False)
    yield
    # Clean up after the test
    _log_stats.clear()
    monkeypatch.setattr('axion._core.logging._logging_configured', False)
    logging.getLogger().handlers.clear()


def test_basic_logging_levels_and_stats(capsys):
    """
    Tests that basic logging calls are emitted and that stats are tracked correctly.
    """
    logger = get_logger('test_logger')
    configure_logging(
        level='DEBUG', use_rich=False, force=True
    )  # Use simple formatter for predictable output

    logger.debug('debug message')
    logger.info('info message')
    logger.warning('warn message')
    logger.error('error message')
    logger.critical('critical message')

    # Check captured output
    captured = capsys.readouterr().err
    assert 'debug message' in captured
    assert 'info message' in captured
    assert 'warn message' in captured
    assert 'error message' in captured
    assert 'critical message' in captured

    # Check that stats were recorded correctly
    stats = _log_stats.get('test_logger', {})
    assert stats.get('DEBUG') == 1
    assert stats.get('INFO') == 1
    assert stats.get('WARNING') == 1
    assert stats.get('ERROR') == 1
    assert stats.get('CRITICAL') == 1


def test_custom_methods_emit_expected_prefixes(capsys):
    """
    Tests the custom logging methods like success(), warning_highlight(), etc.
    """
    logger = get_logger('custom_logger')
    # Using rich=True to ensure emojis are part of the formatted message
    configure_logging(level='INFO', use_rich=False, force=True)

    logger.success('yay!')
    logger.warning_highlight('be careful')
    logger.error_highlight('fail here')

    captured = capsys.readouterr().err
    assert '✅ yay!' in captured
    assert '⚠️  be careful' in captured
    assert '❌ fail here' in captured


def test_log_json_pretty_prints_and_logs(capsys):
    """
    Tests that log_json correctly formats and outputs JSON data.
    """
    logger = get_logger('json_logger')
    configure_logging(level='INFO', use_rich=False, force=True)
    data = {'foo': 'bar', 'baz': [1, 2, 3]}

    logger.log_json(data, title='My JSON')
    captured = capsys.readouterr().err

    assert 'My JSON' in captured
    # Check for a snippet of the pretty-printed JSON
    assert '"foo": "bar"' in captured
    assert '"baz":' in captured

    # Test malformed data triggers error log
    class BadObject:
        def __str__(self):
            raise TypeError('Cannot be serialized')

    logger.log_json({'bad': BadObject()})
    captured_err = capsys.readouterr().err
    assert 'Failed to serialize JSON data' in captured_err


def test_log_operation_context_manager_success(capsys):
    """
    Tests the log_operation context manager on a successful operation.
    """
    logger = get_logger('op_logger')
    configure_logging(level='INFO', use_rich=False, force=True)

    with logger.log_operation('test_op'):
        pass

    captured = capsys.readouterr().err
    assert '🔹 Starting | test_op' in captured
    assert '✅ Completed | test_op in' in captured
    assert '⚡ Performance | test_op | Duration:' in captured


def test_log_operation_context_manager_exception(capsys):
    """
    Tests the log_operation context manager when an exception occurs.
    """
    logger = get_logger('op_logger_fail')
    configure_logging(level='INFO', use_rich=False, force=True)

    with pytest.raises(ValueError, match='fail!'):
        with logger.log_operation('failing_op'):
            raise ValueError('fail!')

    captured = capsys.readouterr().err
    assert '🔹 Starting | failing_op' in captured
    assert '❌ Failed   | failing_op after' in captured


@pytest.mark.asyncio
async def test_async_log_operation_success(capsys):
    """
    Tests the async_log_operation context manager on a successful operation.
    """
    logger = get_logger('async_logger')
    configure_logging(level='INFO', use_rich=False, force=True)

    async with logger.async_log_operation('async_op'):
        await asyncio.sleep(0.01)

    captured = capsys.readouterr().err
    assert '🔹 Starting Async | async_op' in captured
    assert '✅ Completed Async | async_op in' in captured
    assert '⚡ Performance | async_op | Duration:' in captured


@pytest.mark.asyncio
async def test_async_log_operation_exception(capsys):
    """
    Tests the async_log_operation context manager when an exception occurs.
    """
    logger = get_logger('async_logger_fail')
    configure_logging(level='INFO', use_rich=False, force=True)

    with pytest.raises(RuntimeError, match='fail async!'):
        async with logger.async_log_operation('fail_async_op'):
            raise RuntimeError('fail async!')

    captured = capsys.readouterr().err
    assert '🔹 Starting Async | fail_async_op' in captured
    assert '❌ Failed Async | fail_async_op after' in captured


def test_configure_logging_force_reconfigures():
    """
    Tests that `force=True` allows re-configuration.
    """
    # Initial default configuration
    _ = get_logger('reconfig_test')
    assert logging.getLogger().level == logging.getLevelName(DEFAULT_LOG_LEVEL)

    # Force re-configuration to a different level
    configure_logging(level='DEBUG', force=True)
    assert logging.getLogger().level == logging.DEBUG


@pytest.mark.skipif(not RICH_AVAILABLE, reason='rich library not installed')
def test_log_table_prints_table(capsys):
    """
    Tests that log_table() prints a table with Rich.
    """
    logger = get_logger('table_logger')
    configure_logging(level='INFO', use_rich=False, force=True)

    data = [{'name': 'Alice', 'score': 10}, {'name': 'Bob', 'score': 15}]
    logger.log_table(data, title='Scores')

    captured = capsys.readouterr().err
    assert 'Scores' in captured
    assert 'Alice' in captured
    assert '10' in captured
    assert 'Bob' in captured
    assert '15' in captured


def test_log_table_empty_data_logs_message(capsys):
    """
    Tests that log_table() handles empty data gracefully.
    """
    logger = get_logger('table_logger_empty')
    configure_logging(level='INFO', use_rich=False, force=True)

    logger.log_table([], title='Empty Table')
    captured = capsys.readouterr().err
    assert 'Empty Table: No data to display' in captured


def test_log_performance_message_format(capsys):
    """
    Tests the format of the log_performance() message.
    """
    logger = get_logger('perf_logger')
    configure_logging(level='INFO', use_rich=False, force=True)

    logger.log_performance('my_operation', 1.2345, throughput=100, errors=2)
    captured = capsys.readouterr().err

    assert 'my_operation' in captured
    assert 'Duration: 1.2345s' in captured
    assert 'throughput=100' in captured
    assert 'errors=2' in captured


def test_log_exception_logs_error(capsys):
    """
    Tests that log_exception() correctly logs an exception with a traceback.
    """
    logger = get_logger('exc_logger')
    configure_logging(level='ERROR', use_rich=False, force=True)

    try:
        raise ValueError('test error')
    except ValueError as e:
        logger.log_exception(e, 'Custom error message')

    captured = capsys.readouterr().err
    assert 'Custom error message' in captured
    assert 'ValueError: test error' in captured
    # Check for traceback indicator
    assert 'Traceback (most recent call last):' in captured


def test_to_bool_function():
    """
    Tests the _to_bool helper function with various inputs.
    """
    assert _to_bool('true') is True
    assert _to_bool('1') is True
    assert _to_bool('yes') is True
    assert _to_bool('ON') is True
    assert _to_bool('enable') is True
    assert _to_bool('Enabled') is True
    assert _to_bool('false') is False
    assert _to_bool('0') is False
    assert _to_bool('no') is False
    assert _to_bool('other') is False
    assert _to_bool(None) is False
