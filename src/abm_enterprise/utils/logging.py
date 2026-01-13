"""Structured logging configuration using structlog.

Provides consistent, structured logging across all ABM components.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import structlog


def setup_logging(
    level: str | int = "INFO",
    output_dir: Path | str | None = None,
    json_logs: bool = False,
) -> None:
    """Configure structured logging for the application.

    Sets up structlog with appropriate processors for development
    or production use. Optionally writes logs to a file.

    Args:
        level: Log level (e.g., "DEBUG", "INFO", "WARNING", "ERROR").
        output_dir: Optional directory for log file output.
        json_logs: If True, output JSON-formatted logs (for production).

    Example:
        >>> setup_logging(level="DEBUG", output_dir="outputs/logs")
    """
    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )

    # Set up handlers
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        log_file = output_path / "simulation.log"
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(level)
        handlers.append(file_handler)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers = handlers
    root_logger.setLevel(level)

    # Define shared processors
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_logs:
        # Production: JSON output
        structlog.configure(
            processors=[
                *shared_processors,
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
    else:
        # Development: colored console output
        structlog.configure(
            processors=[
                *shared_processors,
                structlog.dev.ConsoleRenderer(colors=True),
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger for a component.

    Args:
        name: Logger name (typically __name__ of the module).

    Returns:
        A bound structlog logger.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Starting simulation", seed=42, country="tanzania")
    """
    return structlog.get_logger(name)


def log_simulation_start(
    logger: structlog.stdlib.BoundLogger,
    run_id: str,
    seed: int,
    country: str,
    scenario: str,
    **kwargs: Any,
) -> None:
    """Log the start of a simulation run.

    Args:
        logger: The logger instance.
        run_id: Unique run identifier.
        seed: Random seed.
        country: Country code.
        scenario: Scenario name.
        **kwargs: Additional key-value pairs to log.
    """
    logger.info(
        "Simulation starting",
        run_id=run_id,
        seed=seed,
        country=country,
        scenario=scenario,
        **kwargs,
    )


def log_simulation_end(
    logger: structlog.stdlib.BoundLogger,
    run_id: str,
    num_steps: int,
    elapsed_seconds: float,
    **kwargs: Any,
) -> None:
    """Log the end of a simulation run.

    Args:
        logger: The logger instance.
        run_id: Unique run identifier.
        num_steps: Number of simulation steps completed.
        elapsed_seconds: Total runtime in seconds.
        **kwargs: Additional key-value pairs to log.
    """
    logger.info(
        "Simulation completed",
        run_id=run_id,
        num_steps=num_steps,
        elapsed_seconds=round(elapsed_seconds, 2),
        **kwargs,
    )
