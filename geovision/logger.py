"""Structured logging configuration for GeoVision."""

import logging
import sys

try:
    from rich.console import Console
    from rich.logging import RichHandler
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False

_DEFAULT_LOGGER: logging.Logger | None = None


def get_logger(name: str = "geovision", level: int = logging.INFO) -> logging.Logger:
    """Get or configure a structured logger with formatting.

    Args:
        name: Logger hierarchy name.
        level: Logging level (e.g. logging.INFO, logging.DEBUG).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    if _HAS_RICH:
        console = Console(file=sys.stdout, color_system="auto")
        handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
            tracebacks_show_locals=False,
        )
        formatter = logging.Formatter("%(message)s")
    else:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
