"""PostgreSQL connection pool for RetailPulse."""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Generator

import psycopg2
import psycopg2.extensions
from psycopg2 import pool

log = logging.getLogger(__name__)

# Cast NUMERIC/DECIMAL → float so psycopg2 never returns decimal.Decimal objects.
# Without this, pages that do arithmetic on PG-sourced columns crash with
# "unsupported operand type(s) for *: 'decimal.Decimal' and 'float'".
_DEC2FLOAT = psycopg2.extensions.new_type(
    psycopg2.extensions.DECIMAL.values,
    "DEC2FLOAT",
    lambda value, _curs: float(value) if value is not None else None,
)
psycopg2.extensions.register_type(_DEC2FLOAT)

_DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres123@localhost:15432/postgres",
)

_pool: pool.ThreadedConnectionPool | None = None


def _get_pool() -> pool.ThreadedConnectionPool | None:
    global _pool
    if _pool is not None:
        return _pool
    try:
        _pool = pool.ThreadedConnectionPool(minconn=1, maxconn=10, dsn=_DATABASE_URL)
        log.info("PostgreSQL pool created")
    except Exception as exc:
        log.warning("PostgreSQL unavailable: %s", exc)
    return _pool


@contextmanager
def get_conn() -> Generator:
    """Yield a psycopg2 connection from the pool; auto-return on exit."""
    p = _get_pool()
    if p is None:
        yield None
        return
    conn = None
    try:
        conn = p.getconn()
        yield conn
        conn.commit()
    except Exception:
        if conn:
            conn.rollback()
        raise
    finally:
        if conn and p:
            p.putconn(conn)


def query_df(sql: str, params=None):
    """
    Run a SELECT and return a pandas DataFrame, or None on DB failure.
    Builds the DataFrame directly from the psycopg2 cursor to avoid
    pandas/SQLAlchemy version-compatibility warnings.
    """
    import pandas as pd

    with get_conn() as conn:
        if conn is None:
            return None
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                cols = [d[0] for d in cur.description]
                rows = cur.fetchall()
            return pd.DataFrame(rows, columns=cols)
        except Exception as exc:
            log.warning("DB query failed: %s", exc)
            return None


def execute(sql: str, params=None) -> bool:
    """Run a non-SELECT statement. Return True on success."""
    with get_conn() as conn:
        if conn is None:
            return False
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
            return True
        except Exception as exc:
            log.warning("DB execute failed: %s", exc)
            return False


def is_available() -> bool:
    """Return True if PostgreSQL is reachable."""
    with get_conn() as conn:
        return conn is not None
