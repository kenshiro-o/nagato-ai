# Standard Library
import logging
from typing import Any, Dict, Optional

# Third Party
import structlog


class StructuredLogger:
    """
    StructuredLogger provides structured logging capabilities using structlog.
    """

    @classmethod
    def get_logger(cls, bound_values: Dict[str, Any], default_log_level: int = logging.INFO) -> structlog.BoundLogger:
        """
        Returns a configured structlog logger instance.
        Creates and configures the logger on first call, then returns cached instance.

        :return: The configured logger instance
        """

        # Configure basic logging
        logging.basicConfig(
            format="%(message)s",
            level=default_log_level,
        )

        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.processors.TimeStamper(fmt="ISO"),
                structlog.processors.JSONRenderer(indent=2, sort_keys=True),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=False,
        )

        logger = structlog.get_logger()
        logger = logger.bind(**bound_values)

        return logger
