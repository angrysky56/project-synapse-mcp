"""
Logging configuration for Project Synapse MCP Server.

Following MCP best practices, all logging goes to stderr to avoid
interfering with protocol operation on stdout.
"""

import logging
import sys
from pathlib import Path


def setup_logging(
    name: str,
    level: str = "INFO",
    log_to_file: bool = False,
    log_file_path: str | None = None
) -> logging.Logger:
    """
    Configure logging for MCP server components.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_to_file: Whether to also log to file
        log_file_path: Path to log file if log_to_file is True

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Clear any existing handlers
    logger.handlers.clear()

    # Set level
    logger.setLevel(getattr(logging, level.upper()))

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the configured settings."""
    return logging.getLogger(name)
