"""
Structured JSON logging configuration for unified backend

Provides comprehensive logging with trace ID correlation, structured JSON format,
and integration with the audit logger for compliance requirements.
"""

import logging
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from contextvars import ContextVar


# Thread-safe trace ID storage using context variables
# This allows trace_id to be set once per request and accessed throughout the call stack
trace_id_var: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)


class StructuredJSONFormatter(logging.Formatter):
    """
    Custom formatter that outputs logs as structured JSON.

    Each log entry contains:
    - timestamp: ISO 8601 format with UTC timezone
    - level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - logger: Logger name (e.g., 'backend.pipeline')
    - trace_id: Request correlation ID (from context)
    - event: Event type (e.g., 'pii_check', 'query_received')
    - message: Human-readable message
    - agent: Which component generated the log (e.g., 'pii_detector', 'router')
    - details: Additional structured data (dict)
    - exception: Stack trace if exception occurred
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "trace_id": trace_id_var.get(),
            "event": getattr(record, "event", "log"),
            "message": record.getMessage(),
        }

        # Add agent name if provided
        if hasattr(record, "agent"):
            log_entry["agent"] = record.agent

        # Add details dict if provided
        if hasattr(record, "details"):
            log_entry["details"] = record.details

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


class HumanReadableFormatter(logging.Formatter):
    """
    Formatter for console output that's easy to read during development.

    Format: timestamp [LEVEL] logger: message
    Example: 2025-01-15 10:30:45 [INFO] backend.pipeline: PII check passed
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record for human readability"""
        # Build base message
        base = f"%(asctime)s [%(levelname)s] %(name)s: %(message)s"

        # Add event if present
        if hasattr(record, "event"):
            base = f"%(asctime)s [%(levelname)s] %(name)s [{record.event}]: %(message)s"

        # Add agent if present
        if hasattr(record, "agent"):
            base = f"%(asctime)s [%(levelname)s] %(name)s ({record.agent}): %(message)s"

        formatter = logging.Formatter(base, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


def setup_logging(log_file: str = "logs/backend.log", level: str = "INFO", json_logs: bool = True):
    """
    Configure structured logging for the application.

    Args:
        log_file: Path to log file
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_logs: If True, use JSON formatter for file; if False, use human-readable

    Returns:
        Configured root logger instance
    """
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # ==================== File Handler (JSON or Human-Readable) ====================
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(getattr(logging, level.upper()))

    if json_logs:
        file_handler.setFormatter(StructuredJSONFormatter())
    else:
        file_handler.setFormatter(HumanReadableFormatter())

    logger.addHandler(file_handler)

    # ==================== Console Handler (Always Human-Readable) ====================
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(HumanReadableFormatter())

    # Configure console encoding for Windows compatibility
    if hasattr(console_handler.stream, 'reconfigure'):
        console_handler.stream.reconfigure(encoding='utf-8', errors='replace')

    logger.addHandler(console_handler)

    return logger


class TraceContext:
    """
    Context manager for trace ID propagation.

    Usage:
        with TraceContext() as trace_id:
            # All log_event() calls within this context will include this trace_id
            log_event(logger, "query_received", "server", {...})

    The trace_id is stored in a context variable, making it available throughout
    the async call stack without explicitly passing it to every function.
    """

    def __init__(self, trace_id: Optional[str] = None):
        """
        Initialize trace context.

        Args:
            trace_id: Optional trace ID. If not provided, generates new UUID.
        """
        self.trace_id = trace_id or str(uuid.uuid4())

    def __enter__(self):
        """Set trace ID in context and return it"""
        trace_id_var.set(self.trace_id)
        return self.trace_id

    def __exit__(self, *args):
        """Clear trace ID from context"""
        trace_id_var.set(None)


def log_event(
    logger: logging.Logger,
    event: str,
    agent: str = "system",
    details: Optional[Dict[str, Any]] = None,
    level: str = "INFO"
):
    """
    Log a structured event.

    This is the primary logging function used throughout the application.
    It automatically includes the trace_id from the current context.

    Args:
        logger: Logger instance to use
        event: Event name/type (e.g., 'pii_check', 'query_received', 'validation_blocked')
        agent: Component that generated the event (e.g., 'pii_detector', 'router', 'pipeline')
        details: Additional structured data to include in the log
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Example:
        log_event(logger, "pii_check", "pii_detector", {
            "safe": False,
            "issues_count": 2,
            "pii_types": ["aadhaar", "pan"]
        }, level="ERROR")

        Output (JSON):
        {
            "timestamp": "2025-01-15T10:30:45.123Z",
            "level": "ERROR",
            "logger": "backend.pipeline",
            "trace_id": "550e8400-e29b-41d4-a716-446655440000",
            "event": "pii_check",
            "message": "pii_check",
            "agent": "pii_detector",
            "details": {
                "safe": false,
                "issues_count": 2,
                "pii_types": ["aadhaar", "pan"]
            }
        }
    """
    extra = {"event": event, "agent": agent}
    if details:
        extra["details"] = details

    log_level = getattr(logging, level.upper())
    logger.log(log_level, event, extra=extra)


# ==================== Utility Functions ====================

def get_current_trace_id() -> Optional[str]:
    """
    Get the current trace ID from context.

    Returns:
        Current trace ID or None if not in a trace context
    """
    return trace_id_var.get()


def set_trace_id(trace_id: str):
    """
    Manually set trace ID in context.

    Useful when receiving trace_id from external source (e.g., HTTP header).

    Args:
        trace_id: Trace ID to set
    """
    trace_id_var.set(trace_id)
