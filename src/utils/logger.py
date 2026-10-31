"""
Logging configuration for MOT tracking pipeline.
"""

import logging
import sys


def get_logger(name: str = "MOT_Tracker", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger with clean console formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

