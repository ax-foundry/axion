import asyncio
import json
import logging
import sys

import pytest

from axion._core.logging import (
    DEFAULT_LOG_LEVEL,
    RICH_AVAILABLE,
    JsonFormatter,
    _log_stats,
    _to_bool,
    clear_logging_config,
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
    configure_logging(level='DEBUG', use_rich=False)
    logger = get_logger('test_logger')

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
    configure_logging(level='INFO', use_rich=False)
    logger = get_logger('custom_logger')

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
    configure_logging(level='INFO', use_rich=False)
    logger = get_logger('json_logger')
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
    configure_logging(level='INFO', use_rich=False)
    logger = get_logger('op_logger')

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
    configure_logging(level='INFO', use_rich=False)
    logger = get_logger('op_logger_fail')

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
    configure_logging(level='INFO', use_rich=False)
    logger = get_logger('async_logger')

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
    configure_logging(level='INFO', use_rich=False)
    logger = get_logger('async_logger_fail')

    with pytest.raises(RuntimeError, match='fail async!'):
        async with logger.async_log_operation('fail_async_op'):
            raise RuntimeError('fail async!')

    captured = capsys.readouterr().err
    assert '🔹 Starting Async | fail_async_op' in captured
    assert '❌ Failed Async | fail_async_op after' in captured


def test_clear_logging_config_allows_reconfiguration():
    """
    Tests that clear_logging_config() allows re-configuration.
    """
    # Initial default configuration
    configure_logging(level=DEFAULT_LOG_LEVEL)
    _ = get_logger('reconfig_test')
    assert logging.getLogger().level == logging.getLevelName(DEFAULT_LOG_LEVEL)

    # Clear and re-configure to a different level
    clear_logging_config()
    configure_logging(level='DEBUG')
    assert logging.getLogger().level == logging.DEBUG


@pytest.mark.skipif(not RICH_AVAILABLE, reason='rich library not installed')
def test_log_table_prints_table(capsys):
    """
    Tests that log_table() prints a table with Rich.
    """
    configure_logging(level='INFO', use_rich=False)
    logger = get_logger('table_logger')

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
    configure_logging(level='INFO', use_rich=False)
    logger = get_logger('table_logger_empty')

    logger.log_table([], title='Empty Table')
    captured = capsys.readouterr().err
    assert 'Empty Table: No data to display' in captured


def test_log_performance_message_format(capsys):
    """
    Tests the format of the log_performance() message.
    """
    configure_logging(level='INFO', use_rich=False)
    logger = get_logger('perf_logger')

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
    configure_logging(level='INFO', use_rich=False)
    logger = get_logger('exc_logger')

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


def test_non_tty_default_emits_json_lines(capsys, monkeypatch):
    """
    Without an explicit choice, a non-interactive stderr gets one JSON object
    per record carrying the keys a log collector promotes.
    """
    monkeypatch.setattr('axion._core.logging._stderr_is_tty', lambda: False)
    monkeypatch.setattr('axion._core.logging.settings.log_use_rich', None)
    monkeypatch.setattr('axion._core.logging.settings.log_json', None)
    configure_logging(level='INFO')
    logger = get_logger('json_logger')

    logger.warning('disk %s full', 'almost', extra={'tenant': 'mgt'})

    lines = [ln for ln in capsys.readouterr().err.splitlines() if ln.strip()]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record['severity'] == 'WARNING'
    assert record['message'] == 'disk almost full'
    assert record['logger'] == 'json_logger'
    assert record['tenant'] == 'mgt'
    assert record['time'].endswith('Z')


def test_tty_default_prefers_rich(monkeypatch):
    """An interactive stderr keeps the rich handler when nothing is configured."""
    monkeypatch.setattr('axion._core.logging._stderr_is_tty', lambda: True)
    monkeypatch.setattr('axion._core.logging.settings.log_use_rich', None)
    monkeypatch.setattr('axion._core.logging.settings.log_json', None)
    configure_logging(level='INFO')

    handler = logging.getLogger().handlers[0]
    if RICH_AVAILABLE:
        assert type(handler).__name__ == 'RichHandler'
    else:
        assert not isinstance(handler.formatter, JsonFormatter)


def test_explicit_rich_off_without_json_keeps_plain_text(capsys, monkeypatch):
    """use_rich=False alone still emits the plain text format, as before."""
    monkeypatch.setattr('axion._core.logging._stderr_is_tty', lambda: False)
    monkeypatch.setattr('axion._core.logging.settings.log_json', None)
    configure_logging(level='INFO', use_rich=False)
    get_logger('plain_logger').info('plain message')

    captured = capsys.readouterr().err
    assert 'plain message' in captured
    assert 'plain_logger' in captured
    assert not captured.lstrip().startswith('{')


def test_json_formatter_folds_exception_text():
    """A traceback lands under ``exception`` and the line stays one JSON object."""
    formatter = JsonFormatter()
    try:
        raise ValueError('boom')
    except ValueError:
        record = logging.LogRecord(
            'x', logging.ERROR, __file__, 1, 'failed', None, sys.exc_info()
        )
    out = json.loads(formatter.format(record))
    assert out['severity'] == 'ERROR'
    assert out['message'] == 'failed'
    assert 'ValueError: boom' in out['exception']
    assert '\n' not in formatter.format(record)
