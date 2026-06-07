"""Audit log — PostgreSQL primary, JSON-lines file fallback."""
from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

_LOG_DIR  = Path(__file__).resolve().parents[2] / "logs"
_LOG_FILE = _LOG_DIR / "audit.log"


def log_action(
    username: str,
    action: str,
    resource: str,
    details: str = "",
) -> None:
    ts    = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    entry = {"ts": ts, "user": username, "action": action, "resource": resource, "details": details}

    # 1. PostgreSQL
    try:
        from utils.db import get_conn
        with get_conn() as conn:
            if conn is not None:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO audit_log (ts, username, action, resource, details) "
                        "VALUES (%s, %s, %s, %s, %s)",
                        (ts, username, action, resource, details),
                    )
                return
    except Exception as exc:
        log.warning("Audit DB write failed, using file fallback: %s", exc)

    # 2. File fallback
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    with _LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def read_log(limit: int = 500) -> list[dict]:
    # 1. PostgreSQL
    try:
        from utils.db import query_df
        df = query_df(
            "SELECT ts, username AS \"user\", action, resource, details "
            "FROM audit_log ORDER BY ts DESC LIMIT %s",
            params=(limit,),
        )
        if df is not None:
            return df.to_dict("records")
    except Exception as exc:
        log.warning("Audit DB read failed, using file fallback: %s", exc)

    # 2. File fallback
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
