"""
Logging configuration for Project Synapse MCP Server.

Following MCP best practices, all logging goes to stderr to avoid
interfering with protocol operation on stdout.
"""

import functools
import inspect
import logging
import sys
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar

from .metrics import metrics

T = TypeVar("T")


class SynapseLogger:
    """
    Enhanced logger for Synapse MCP components.

    Supports standard logging to stderr, performance timing, and
    optional status updates to the MCP client via the Context object.
    """

    def __init__(self, name: str, level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        self._ctx: Any = None

    @property
    def name(self) -> str:
        """Return the name of the underlying logger."""
        return self.logger.name

    def set_context(self, ctx: Any) -> None:
        """Set the MCP Context for status updates."""
        self._ctx = ctx

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        """Log info to stderr and optionally send status to MCP Context."""
        self.logger.info(msg, *args, **kwargs)
        if self._ctx and hasattr(self._ctx, "info"):
            try:
                # Format message if args are provided
                formatted_msg = msg % args if args else msg
                # info() is a sync call in FastMCP.Context
                self._ctx.info(formatted_msg)
            except Exception:  # pylint: disable=broad-exception-caught
                pass

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.error(msg, *args, **kwargs)

    def exception(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self.logger.exception(msg, *args, **kwargs)

    @contextmanager
    def timed(self, operation: str) -> Generator[None, None, None]:
        """Context manager to time an operation and log its duration."""
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.logger.info("Operation '%s' took %.4f seconds", operation, duration)
            metrics.record(operation, duration)

    def timer(
        self, operation: str | None = None
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator to time a function call."""

        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            op_name = operation or func.__name__

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> T:
                with self.timed(op_name):
                    return func(*args, **kwargs)

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> T:
                with self.timed(op_name):
                    return await func(*args, **kwargs)  # type: ignore

            if inspect.iscoroutinefunction(func):
                return async_wrapper  # type: ignore
            return wrapper

        return decorator


def setup_logging(
    name: str,
    level: str = "INFO",
    log_to_file: bool = False,
    log_file_path: str | None = None,
) -> SynapseLogger:
    """
    Configure logging for MCP server components.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_to_file: Whether to also log to file
        log_file_path: Path to log file if log_to_file is True

    Returns:
        SynapseLogger instance
    """
    logger = logging.getLogger(name)

    # Clear any existing handlers
    logger.handlers.clear()

    # Set level
    logger.setLevel(getattr(logging, level.upper()))

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Always log to stderr for MCP compliance
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)

    # Optionally log to file as well
    if log_to_file and log_file_path:
        log_path = Path(log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    synapse_logger = SynapseLogger(name, level)
    _loggers[name] = synapse_logger
    return synapse_logger


_loggers: dict[str, SynapseLogger] = {}


def get_logger(name: str) -> SynapseLogger:
    """Get a SynapseLogger instance."""
    if name not in _loggers:
        _loggers[name] = SynapseLogger(name)
    return _loggers[name]
