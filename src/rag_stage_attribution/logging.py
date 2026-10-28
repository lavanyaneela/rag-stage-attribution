"""Structured logging that stays out of measured output.

Experiments write machine readable JSON to files; logs are for a human
watching a run. Keeping the two channels separate is what lets the receipts
mechanism treat every committed experiment file as data.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any


class JsonFormatter(logging.Formatter):
    """One JSON object per line, stable key order, no timestamps by default.

    Timestamps are omitted so that two runs of the same command produce the
    same log bytes; determinism is a project rule, not just a test rule.
    """

    def __init__(self, with_time: bool = False) -> None:
        super().__init__()
        self.with_time = with_time

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname.lower(),
            "logger": record.name,
            "event": record.getMessage(),
        }
        extra = getattr(record, "fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if self.with_time:
            payload["time"] = self.formatTime(record)
        return json.dumps(payload, sort_keys=True)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    logger.info(event, extra={"fields": fields})
