# AXION Logging

The `axion` logging module is a flexible system designed to provide a rich, informative logging experience. It intelligently adapts its behavior based on its environment, working seamlessly as a standalone application or as a plugin within a host system like MLPTK.

---

## Core Concepts

The logging system operates in two distinct modes, which it switches between automatically based on the `AXION_MODE` environment variable.

### 1. Standalone Mode

When `AXION_MODE` is set to `standalone`, the logger provides a rich, developer-friendly experience directly in your console using the **`rich`** library.

- **Beautiful Output:** Get colorized log levels, emojis, and beautifully formatted tables and tracebacks.
- **High-Level Methods:** Use convenient methods like `logger.success()`, `logger.log_table()`, and the `logger.log_operation()` context manager.
- **Zero-Fuss Default:** This rich output is the default in standalone mode.

### 2. Plugin Mode

When `AXION_MODE` is set to `plugin` (or is unset), the logger's behavior changes to integrate with the host application.

- **Transparent Redirection:** Instead of printing to the console itself, it transparently **forwards** all log messages to the host application's logger.
- **Unified Logs:** This ensures the logs are perfectly integrated with the host's logs, using the host's formatting, colors, and output destinations (e.g., files, external services).
- **Automatic Activation:** This is the default mode. The host application is responsible for the final logging configuration.

---

## Basic Usage

For most use cases, you can simply import the global `logger` instance. For larger applications, it's best practice to get a logger specific to your module.

```python
# Best Practice: Get a module-specific logger
from axion.core.logging import get_logger

# The __name__ variable automatically gets the current module's name
logger = get_logger(__name__)

logger.warning("This warning came from the '%s' module.", __name__)
logger.success("A task completed successfully!")
```

## Configuration

Configuration is primarily handled by your Pydantic settings object but can be overridden by passing arguments directly to the `configure_logging` function.

### Pydantic Settings & Environment Variables

The following settings are read from your global settings object, which is populated from environment variables (e.g., `AXION_LOG_LEVEL`) or your `.env` file.

| Setting | Environment Variable | Description |
|---|---|---|
| `log_level` | `AXION_LOG_LEVEL` | The minimum logging level (e.g., 'DEBUG'). |
| `log_use_rich` | `AXION_LOG_USE_RICH` | Use rich for beautiful console output. |
| `log_format_string` | `AXION_LOG_FORMAT_STRING` | A custom format string for the logger. |
| `log_file_path` | `AXION_LOG_FILE_PATH` | If set, logs will be written to this file. |

### Programmatic Configuration

Call `configure_logging` once at application startup.

#### Example 1: Using Global Settings

This is the simplest and most common configuration. It reads all settings from your Pydantic settings object.

```python
from axion.core.logging import configure_logging, logger

# Reads level, rich status, etc., from the global settings
configure_logging()

logger.info("Using the configuration from the environment.")
```

#### Example 2: Overriding Global Settings

You can pass arguments to `configure_logging` to override the values from the settings object for a specific session. This is especially useful for testing.

```python
from axion.core.logging import configure_logging, logger

# This will force the log level to DEBUG for this session,
# ignoring the value in the settings object.
configure_logging(level="DEBUG", force=True)

logger.debug("This debug message is now visible.")
```

> **Note:** The logger is designed to be configured only once. If you need to change the configuration after it has been set, you must include the `force=True` flag.

## Additional Logging Methods

The RichLogger provides several high-level methods to make your logs more expressive.

| Method | Description |
|---|---|
| `logger.success(msg)` | Logs an info-level message with a ✅ emoji. |
| `logger.warning_highlight(msg)` | Logs a warning-level message with a ⚠️ emoji. |
| `logger.error_highlight(msg)` | Logs an error-level message with a ❌ emoji. |
| `logger.log_table(data)` | Prints a list of dictionaries as a formatted table. |
| `logger.log_json(data)` | Pretty-prints a dictionary or list as JSON. |
| `logger.log_operation(name)` | A context manager that logs the start, end, and duration of a code block. |

### Example

```python
from axion.core.logging import logger
import time

users = [
    {"id": "user_1", "status": "active", "last_login": "2025-08-02"},
    {"id": "user_2", "status": "inactive", "last_login": "2025-07-15"},
]
logger.log_table(users, title="User Status")

with logger.log_operation("Process User Data"):
    # Your code here...
    time.sleep(0.5)
```

## Advanced Usage

### Module-Specific Loggers

For better organization and debugging, create module-specific loggers:

```python
from axion.core.logging import get_logger

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

### Error Handling with Rich Logging

```python
from axion.core.logging import logger

try:
    risky_operation()
except Exception as e:
    logger.error_highlight(f"Operation failed: {e}")
    # Rich automatically formats the traceback beautifully
    logger.exception("Full traceback:")
```

### Conditional Logging

```python
from axion.core.logging import logger

def debug_intensive_operation(data):
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Processing {len(data)} items")
        logger.log_json(data[:5])  # Log first 5 items as JSON
```

## Best Practices

1. **Use Module-Specific Loggers**: Always use `get_logger(__name__)` for better log organization.

2. **Configure Once**: Call `configure_logging()` only once at application startup.

3. **Use Context Managers**: Leverage `logger.log_operation()` for timing operations.

4. **Rich Methods in Development**: Use `logger.success()`, `logger.log_table()` etc. for better development experience.

5. **Respect the Environment**: Let the logging system automatically adapt to standalone vs plugin mode.

```python
# ✅ Good
from axion.core.logging import get_logger
logger = get_logger(__name__)

# ✅ Good - At startup
configure_logging()

# ✅ Good - Using context manager
with logger.log_operation("Data Migration"):
    migrate_users()
    migrate_products()

# ❌ Avoid - Configuring multiple times
configure_logging()
# ... later in code ...
configure_logging()  # This will be ignored unless force=True
```
