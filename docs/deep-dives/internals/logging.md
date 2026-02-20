---
icon: custom/terminal
---
# Axion Logging

The Axion logging module provides a rich, informative logging experience with beautiful console output powered by the **rich** library.

---

## Core Features

- **Beautiful Output**: Colorized log levels, emojis, and formatted tables and tracebacks via the `rich` library
- **High-Level Methods**: Convenient methods like `logger.success()`, `logger.log_table()`, and the `logger.log_operation()` context manager
- **Auto-Configuration**: Just use `get_logger()` - it auto-configures from environment variables on first use
- **File Logging**: Optionally write logs to a file in addition to the console

---

## Basic Usage

For most use cases, simply import the global `logger` instance or get a module-specific logger:

```python
# Option 1: Use the global logger
from axion.logging import logger

logger.info("Hello from Axion!")
logger.success("Task completed successfully!")

# Option 2: Get a module-specific logger (recommended)
from axion.logging import get_logger

logger = get_logger(__name__)
logger.warning("This warning came from the '%s' module.", __name__)
```

## Configuration

Configuration is handled via environment variables or by calling `configure_logging()` directly.

### Environment Variables

| Setting | Environment Variable | Default | Description |
|---------|---------------------|---------|-------------|
| `log_level` | `LOG_LEVEL` | `'INFO'` | Minimum logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`). |
| `log_use_rich` | `LOG_USE_RICH` | `True` | Use rich for beautiful console output. |
| `log_format_string` | `LOG_FORMAT_STRING` | `None` | A custom format string for the logger. |
| `log_file_path` | `LOG_FILE_PATH` | `None` | If set, logs will be written to this file. |

### Programmatic Configuration

Call `configure_logging()` once at application startup, or let it auto-configure on first use.

=== ":material-auto-fix: Zero-Config (Recommended)"

    Just use `get_logger()` - it auto-configures from environment variables on first use.

    ```python
    from axion.logging import get_logger

    # Auto-configures from LOG_LEVEL, LOG_USE_RICH, etc.
    logger = get_logger(__name__)
    logger.info("Using the configuration from the environment.")
    ```

=== ":material-cog: Explicit Configuration"

    Call `configure_logging()` before getting loggers to override defaults.

    ```python
    from axion.logging import configure_logging, get_logger

    # Configure first, then get logger
    configure_logging(level="DEBUG", use_rich=True)
    logger = get_logger(__name__)

    logger.debug("This debug message is now visible.")
    ```

=== ":material-refresh: Reconfiguration"

    Call `configure_logging()` with new parameters - it applies immediately.

    ```python
    from axion.logging import configure_logging, get_logger

    logger = get_logger(__name__)  # Auto-configures to INFO

    # Change level on the fly
    configure_logging(level="DEBUG")  # Now DEBUG
    configure_logging(level="ERROR")  # Now ERROR
    ```

!!! note
    Calling `configure_logging()` with explicit parameters always applies them. Calling with no parameters skips if already configured.

## RichLogger Methods

The `RichLogger` class provides several high-level methods to make your logs more expressive.

| Method | Description |
|--------|-------------|
| `logger.success(msg)` | Logs an info-level message with a checkmark emoji. |
| `logger.warning_highlight(msg)` | Logs a warning-level message with a warning emoji. |
| `logger.error_highlight(msg)` | Logs an error-level message with a cross emoji. |
| `logger.log_table(data)` | Prints a list of dictionaries as a formatted table. |
| `logger.log_json(data)` | Pretty-prints a dictionary or list as JSON. |
| `logger.log_performance(operation, duration)` | Logs performance metrics for an operation. |
| `logger.log_exception(e)` | Logs an exception with traceback. |
| `logger.log_operation(name)` | Context manager that logs start, end, and duration of a code block. |
| `logger.async_log_operation(name)` | Async context manager for logging operations. |

### Examples

#### Logging Tables

```python
from axion.logging import logger

users = [
    {"id": "user_1", "status": "active", "last_login": "2025-08-02"},
    {"id": "user_2", "status": "inactive", "last_login": "2025-07-15"},
]
logger.log_table(users, title="User Status")
```

#### Timing Operations

```python
from axion.logging import logger
import time

with logger.log_operation("Process User Data"):
    # Your code here...
    time.sleep(0.5)

# Output:
# 🔹 Starting | Process User Data
# ✅ Completed | Process User Data in 0.50s
```

#### Async Operations

```python
from axion.logging import logger

async def fetch_data():
    async with logger.async_log_operation("Fetch Remote Data"):
        # Your async code here...
        await some_async_operation()
```

#### Logging JSON

```python
from axion.logging import logger

config = {"model": "gpt-4o", "temperature": 0.7}
logger.log_json(config, title="Model Configuration")
```

## Advanced Usage

=== ":material-folder-multiple: Module-Specific Loggers"

    For better organization and debugging, create module-specific loggers:

    ```python
    from axion.logging import get_logger

    # Each module gets its own logger
    logger = get_logger(__name__)

    class MyService:
        def __init__(self):
            self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

        def process_data(self):
            self.logger.info("Starting data processing...")
            # Processing logic here
            self.logger.success("Data processing completed!")
    ```

=== ":material-alert-circle: Error Handling"

    ```python
    from axion.logging import logger

    try:
        risky_operation()
    except Exception as e:
        logger.error_highlight(f"Operation failed: {e}")
        # Rich automatically formats the traceback beautifully
        logger.exception("Full traceback:")
    ```

=== ":material-toggle-switch: Conditional Logging"

    ```python
    import logging
    from axion.logging import logger

    def debug_intensive_operation(data):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Processing {len(data)} items")
            logger.log_json(data[:5])  # Log first 5 items as JSON
    ```

=== ":material-chart-box: Summary"

    Get a summary of all logging activity during the session:

    ```python
    from axion.logging import log_summary

    # At the end of your application
    log_summary()

    # Output:
    # --- Logging Summary ---
    # Total Session Runtime: 45.32 seconds
    # Logger 'axion': 25 messages
    #     - INFO: 20
    #     - WARNING: 3
    #     - ERROR: 2
    # Grand Total Messages: 25
    # -----------------------
    ```

## Configuration State Management

```python
from axion.logging import (
    configure_logging,
    is_logging_configured,
    clear_logging_config
)

# Check if logging is configured
if is_logging_configured():
    print("Logging already set up")

# Full reset if needed
clear_logging_config()
configure_logging(level="DEBUG")
```

## Best Practices

1. **Use Module-Specific Loggers**: Always use `get_logger(__name__)` for better log organization.

2. **Let It Auto-Configure**: Just use `get_logger()` - it configures automatically from environment variables.

3. **Configure Before Use**: If you need custom settings, call `configure_logging()` before `get_logger()`.

4. **Use Context Managers**: Leverage `logger.log_operation()` for timing operations.

5. **Rich Methods in Development**: Use `logger.success()`, `logger.log_table()` etc. for better development experience.

```python
# Good - Zero config, auto-configures from env vars
from axion.logging import get_logger
logger = get_logger(__name__)

# Good - Configure before getting logger
from axion.logging import configure_logging, get_logger
configure_logging(level="DEBUG")
logger = get_logger(__name__)

# Good - Using context manager
with logger.log_operation("Data Migration"):
    migrate_users()
    migrate_products()

# Good - Reconfigure on the fly
from axion.logging import configure_logging
configure_logging(level="DEBUG")  # Always applies with explicit params

# Good - Check configuration state
from axion.logging import is_logging_configured, clear_logging_config
if is_logging_configured():
    clear_logging_config()  # Full reset if needed
```

## API Reference

### Functions

| Function | Description |
|----------|-------------|
| `get_logger(name)` | Get a logger instance, auto-configuring on first use. |
| `configure_logging(level, use_rich, format_string, file_path)` | Configure the logging system. |
| `is_logging_configured()` | Check if logging has been configured. |
| `clear_logging_config()` | Clear the logging configuration and reset state. |
| `log_summary()` | Log a summary of all logging activity during the session. |

### Classes

| Class | Description |
|-------|-------------|
| `RichLogger` | Custom logger class with enhanced logging methods. |
| `LoguruHandler` | Bridge handler for forwarding logs to loguru (if available). |

---

<div class="ref-nav" markdown="1">

[Environment & Settings :octicons-arrow-right-24:](environment.md){ .md-button .md-button--primary }
[Installation :octicons-arrow-right-24:](../../getting-started/installation.md){ .md-button }

</div>
