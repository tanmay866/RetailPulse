"""Redis-backed cache for RetailPulse data loaders."""
from __future__ import annotations

import logging
import os
import pickle
from typing import Any, Callable

import redis as _redis

log = logging.getLogger(__name__)

_REDIS_URL   = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
_DEFAULT_TTL = 3600  # seconds

_client: _redis.Redis | None = None
_unavailable: bool = False  # set True after first failed connect; stops repeated warnings


def _get_client() -> _redis.Redis | None:
    global _client, _unavailable
    if _client is not None:
        return _client
    if _unavailable:
        return None
    try:
        r = _redis.Redis.from_url(
            _REDIS_URL,
            socket_connect_timeout=2,
            socket_timeout=2,
            decode_responses=False,
        )
        r.ping()
        _client = r
        log.info("Redis connected at %s", _REDIS_URL)
    except Exception as exc:
        log.warning("Redis unavailable — cache disabled: %s", exc)
        _unavailable = True
    return _client


def get(key: str) -> Any | None:
    r = _get_client()
    if r is None:
        return None
    try:
        raw = r.get(key)
        return pickle.loads(raw) if raw is not None else None
    except Exception as exc:
        log.warning("Redis GET failed [%s]: %s", key, exc)
        return None


def set(key: str, value: Any, ttl: int = _DEFAULT_TTL) -> None:
    r = _get_client()
    if r is None:
        return
    try:
        r.setex(key, ttl, pickle.dumps(value))
    except Exception as exc:
        log.warning("Redis SET failed [%s]: %s", key, exc)


def delete(key: str) -> None:
    r = _get_client()
    if r is None:
        return
    try:
        r.delete(key)
    except Exception:
        pass


def flush_all() -> None:
    """Invalidate the entire RetailPulse cache namespace."""
    r = _get_client()
    if r is None:
        return
    try:
        keys = r.keys("rp:*")
        if keys:
            r.delete(*keys)
    except Exception as exc:
        log.warning("Redis flush failed: %s", exc)


def cached(key: str, loader: Callable, ttl: int = _DEFAULT_TTL) -> Any:
    """Return cached value, or call loader(), cache it, and return it."""
    value = get(key)
    if value is not None:
        return value
    value = loader()
    if value is not None:
        set(key, value, ttl)
    return value


def is_available() -> bool:
    """Return True if Redis is reachable."""
    return _get_client() is not None
