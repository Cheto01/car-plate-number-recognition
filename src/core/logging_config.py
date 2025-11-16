"""Logging configuration for ALPR System."""
import logging
import logging.config
import sys
from pathlib import Path
from typing import Optional

from loguru import logger

from src.core.config import get_settings


class InterceptHandler(logging.Handler):
    """
    Intercept standard logging messages and redirect to Loguru.
    This allows us to use Loguru while maintaining compatibility with
    libraries that use standard logging.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record through Loguru."""
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where the logged message originated
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging(log_level: Optional[str] = None) -> None:
    """
    Setup logging configuration using Loguru.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    settings = get_settings()
    log_level = log_level or settings.log_level

    # Ensure logs directory exists
    Path(settings.logs_dir).mkdir(parents=True, exist_ok=True)

    # Remove default logger
    logger.remove()

    # Add console logger with color
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True,
    )

    # Add file logger with rotation
    logger.add(
        settings.logs_dir / "alpr.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="INFO",
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        enqueue=True,  # Thread-safe
    )

    # Add error file logger
    logger.add(
        settings.logs_dir / "errors.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="ERROR",
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )

    # Intercept standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # Set levels for noisy libraries
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    logger.info(f"Logging configured with level: {log_level}")


def get_logger(name: str) -> "logger":
    """
    Get a logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logger.bind(name=name)
