"""Append-only JSON-lines audit log for security-relevant events."""
from __future__ import annotations

import datetime
import json
from pathlib import Path

_LOG_DIR  = Path(__file__).resolve().parents[2] / "logs"
_LOG_FILE = _LOG_DIR / "audit.log"


def log_action(
    username: str,
    action: str,
    resource: str,
    details: str = "",
) -> None:
    """Append one entry to logs/audit.log (creates the file if absent)."""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts":       datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "user":     username,
        "action":   action,
        "resource": resource,
        "details":  details,
    }
    with _LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def read_log(limit: int = 500) -> list[dict]:
    """Return the last *limit* entries, newest first."""
    if not _LOG_FILE.exists():
        return []
    lines = _LOG_FILE.read_text(encoding="utf-8").splitlines()
    entries: list[dict] = []
    for line in reversed(lines[-limit:]):
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries
