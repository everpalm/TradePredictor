# Contents of unit/log_handler.py
'''Copyright (c) 2024 Jaron Cheng'''
import logging


def get_logger(module_name: str, level: int = logging.DEBUG) -> logging.Logger:
    """
    Configures and returns a logger for the specified module.

    Args:
        module_name (str): The name of the module requesting the logger.
        level (int): The logging level (default: logging.DEBUG).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)

    # Create a console handler for the logger
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)  # Default handler level

    # Define a formatter and attach it to the handler
    formatter = logging.Formatter(
        # "%(asctime)s %(name)s %(levelname)s %(message)s"
        # "%(asctime)s %(levelname)s %(message)s"
        "%(asctime)s %(levelname)s %(filename)s %(message)s"
    )
    console_handler.setFormatter(formatter)

    # Attach the handler to the logger
    if not logger.handlers:  # Prevent duplicate handlers
        logger.addHandler(console_handler)

    # Avoid propagating logs to the root logger
    logger.propagate = False

    return logger
