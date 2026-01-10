import json
import logging
import sys
import time
from contextlib import asynccontextmanager, contextmanager
from logging import StreamHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from axion._core.environment import settings
from axion._core.utils import Timer

try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    # This is for MLPTK integration
    from loguru import logger as loguru_logger

    LOGURU_AVAILABLE = True
except ImportError:
    LOGURU_AVAILABLE = False

# --- Global State ---
_log_stats: Dict[str, Dict[str, int]] = {}
_session_start_time: float = time.monotonic()
_logging_configured = False
DEFAULT_LOG_LEVEL = 'INFO'


def _to_bool(value: Union[str, bool]) -> bool:
    """Convert string or bool to boolean, handling common variations."""
    if isinstance(value, bool):
        return value
    return value is not None and value.lower() in {
        'true',
        '1',
        'yes',
        'on',
        'enable',
        'enabled',
    }


#########################################
####### Bridge Handler for Loguru #######
#########################################
class LoguruHandler(logging.Handler):
    """
    A logging handler that forwards records from Python's standard logging
    to the loguru logger, making them compatible.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """
        Forwards a log record to loguru. It translates the log level and
        ensures the call site is correctly identified by loguru.
        """
        try:
            level = loguru_logger.level(record.levelname).name
        except ValueError:
            level = str(record.levelno)

        # This depth calculation helps loguru find the original caller
        # by stepping out of the standard logging library's internal frames.
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        loguru_logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


class RichLogger(logging.Logger):
    """
    A custom logger class that inherits from logging.Logger and includes
    a suite of powerful, high-level logging methods.
    """

    def _log(
        self,
        level,
        msg,
        args,
        exc_info=None,
        extra=None,
        stack_info=False,
        stacklevel=1,
        **kwargs,
    ):
        """Override internal _log to track stats before passing to parent."""
        logger_name = self.name
        if logger_name not in _log_stats:
            _log_stats[logger_name] = {
                'DEBUG': 0,
                'INFO': 0,
                'WARNING': 0,
                'ERROR': 0,
                'CRITICAL': 0,
            }
        level_name = logging.getLevelName(level)
        if level_name in _log_stats[logger_name]:
            _log_stats[logger_name][level_name] += 1
        super()._log(level, msg, args, exc_info, extra, stack_info, stacklevel)

    def success(self, message: str, *args, **kwargs):
        """Log a 'SUCCESS' message with a checkmark."""
        self.info(f'✅ {message}', *args, **kwargs)

    def warning_highlight(self, message: str, *args, **kwargs):
        """Log a 'WARNING' message with a warning sign."""
        self.warning(f'⚠️  {message}', *args, **kwargs)

    def error_highlight(self, message: str, *args, **kwargs):
        """Log an 'ERROR' message with a cross mark."""
        self.error(f'❌ {message}', *args, **kwargs)

    def log_table(
        self,
        data: List[Dict[str, Any]],
        title: Optional[str] = None,
        columns: Optional[List[str]] = None,
        level: Union[int, str] = logging.INFO,
    ) -> None:
        """Logs a list of dictionaries as a formatted table string."""
        if not RICH_AVAILABLE:
            self.log_json(data, level=level, title=title or 'Table Data')
            return

        if not data:
            self.log(level, f"{title or 'Table'}: No data to display")
            return

        # Capture the Rich table to a string instead of printing directly
        console = Console(file=sys.stdout, record=True, width=120)
        table = Table(title=title, show_header=True, header_style='bold magenta')
        columns = columns or list(data[0].keys())
        for col in columns:
            table.add_column(str(col), style='cyan', no_wrap=False)
        for row in data:
            values = [str(row.get(col, '')) for col in columns]
            table.add_row(*values)

        console.print(table)
        table_str = console.export_text(clear=True)
        # Log the captured string so it can be handled by any handler
        self.log(level, f'\n{table_str}')

    def log_performance(self, operation: str, duration: float, **metrics: Any):
        """Log performance metrics for a specific operation."""
        msg = f'⚡ Performance | {operation} | Duration: {duration:.4f}s'
        if metrics:
            metric_str = ' | '.join(f'{k}={v}' for k, v in metrics.items())
            msg += f' | {metric_str}'
        self.info(msg)

    def log_json(
        self,
        data: Union[Dict, List],
        level: Union[int, str] = logging.INFO,
        title: Optional[str] = None,
    ):
        """Pretty-print and log a JSON object or list."""
        level_num = (
            getattr(logging, level.upper(), logging.INFO)
            if isinstance(level, str)
            else level
        )
        try:
            json_str = json.dumps(data, indent=2, default=str)
            if title:
                self.log(level_num, f'{title}:')
            self.log(level_num, f'\n{json_str}')
        except Exception as e:
            self.error(f'Failed to serialize JSON data: {e}')

    def log_exception(self, e: Exception, message: Optional[str] = None):
        """Log an exception with traceback."""
        msg = message or f'Exception occurred: {type(e).__name__}: {e}'
        self.error(msg, exc_info=True)

    @contextmanager
    def log_operation(self, operation_name: str, level: int = logging.INFO):
        """Context manager for logging the start, end, and duration of an operation."""
        if self.isEnabledFor(level):
            self.log(level, f'🔹 Starting | {operation_name}')
            timer = Timer()
            timer.start()
            try:
                yield timer
            except Exception as e:
                timer.stop()
                self.error(
                    f'❌ Failed   | {operation_name} after {timer.elapsed_time:.2f}s',
                    exc_info=e,
                )
                raise
            else:
                timer.stop()
                self.log(
                    level,
                    f'✅ Completed | {operation_name} in {timer.elapsed_time:.2f}s',
                )
                self.log_performance(operation_name, timer.elapsed_time)
        else:
            yield Timer()

    @asynccontextmanager
    async def async_log_operation(self, operation_name: str, level: int = logging.INFO):
        """Async context manager for logging an operation."""
        if self.isEnabledFor(level):
            self.log(level, f'🔹 Starting Async | {operation_name}')
            timer = Timer()
            timer.start()
            try:
                yield timer
            except Exception as e:
                timer.stop()
                self.error(
                    f'❌ Failed Async | {operation_name} after {timer.elapsed_time:.2f}s',
                    exc_info=e,
                )
                raise
            else:
                timer.stop()
                self.log(
                    level,
                    f'✅ Completed Async | {operation_name} in {timer.elapsed_time:.2f}s',
                )
                self.log_performance(operation_name, timer.elapsed_time)
        else:
            yield Timer()


def configure_logging(
    level: Optional[str] = None,
    use_rich: Optional[bool] = None,
    format_string: Optional[str] = None,
    file_path: Optional[str] = None,
    force: bool = False,
    **kwargs,
) -> None:
    """
    Configures the application's logging system.

    Prioritizes direct function arguments over global settings, which in turn
    are loaded from environment variables or .env files.

    Args:
        level: Override the log level (e.g., 'DEBUG').
        use_rich: Override the use of rich formatting.
        format_string: Override the log format string.
        file_path: Override the log file path.
        force: If True, will overwrite an existing configuration.
    """
    global _logging_configured, _session_start_time
    init_logger = logging.getLogger(__name__)

    if _logging_configured and not force:
        init_logger.debug('Logging already configured. Skipping reconfiguration.')
        return

    if not _logging_configured:
        _session_start_time = time.monotonic()

    # 1. Resolve configuration, prioritizing direct arguments over global settings.
    final_level = level or settings.log_level
    # A specific check for `is not None` is needed for bools, as `False` is a valid override.
    final_use_rich = use_rich if use_rich is not None else settings.log_use_rich
    final_format_string = format_string or settings.log_format_string
    final_file_path = file_path or settings.log_file_path

    # 2. Configure the root logger
    logging.setLoggerClass(RichLogger)
    root_logger: RichLogger = logging.getLogger()
    root_logger.setLevel(final_level.upper())

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # 3. Standalone Configuration
    init_logger.debug(
        f'--- Configuring standalone logging. Level: {final_level}, Rich: {final_use_rich} ---'
    )
    handler = StreamHandler()
    formatter = None

    if final_use_rich and RICH_AVAILABLE:
        handler = RichHandler(rich_tracebacks=True, show_path=False, markup=True)
        formatter = logging.Formatter('%(message)s', datefmt='[%X]')
    elif final_format_string:
        formatter = logging.Formatter(final_format_string)
    else:
        formatter = logging.Formatter(
            '%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s'
        )

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Configure file handler if a path is provided
    if final_file_path:
        try:
            Path(final_file_path).parent.mkdir(parents=True, exist_ok=True)
            file_formatter_str = (
                final_format_string
                or '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
            )
            file_formatter = logging.Formatter(file_formatter_str)

            file_handler = logging.FileHandler(final_file_path, mode='a')
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
            init_logger.debug(f'Logging also configured for file: {final_file_path}')
        except Exception as e:
            root_logger.error(
                f'Failed to configure file handler at {final_file_path}: {e}'
            )

    _logging_configured = True


def get_logger(name: str) -> RichLogger:
    """
    Gets a logger instance. If logging is not yet configured,
    it applies a safe default configuration first.
    """
    global _logging_configured
    if not _logging_configured:
        configure_logging()
    return logging.getLogger(name)


def log_summary():
    """Logs a summary of all logging activity during the session."""
    logger = get_logger('LoggingSummary')
    total_runtime = time.monotonic() - _session_start_time
    logger.info('--- Logging Summary ---')
    logger.info(f'Total Session Runtime: {total_runtime:.2f} seconds')
    grand_total = sum(sum(stats.values()) for stats in _log_stats.values())
    for logger_name, stats in _log_stats.items():
        total = sum(stats.values())
        if total > 0:
            logger.info(f"Logger '{logger_name}': {total} messages")
            for level, count in stats.items():
                if count > 0:
                    logger.info(f'    - {level}: {count}')
    logger.info(f'Grand Total Messages: {grand_total}')
    logger.info('-----------------------')


# CONVENIENCE ACCESS
logger: RichLogger = get_logger('axion')


__all__ = [
    'configure_logging',
    'get_logger',
    'setup_plugin_logging',
    'log_summary',
    'RichLogger',
    'LoguruHandler',
    'logger',
]
