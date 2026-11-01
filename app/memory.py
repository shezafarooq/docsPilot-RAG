"""Persistent local checkpoints and one active request per conversation."""

import sqlite3
from contextlib import contextmanager
from functools import lru_cache
from threading import Lock

from langgraph.checkpoint.sqlite import SqliteSaver

from app.config import get_settings
from app.errors import ServiceError

_connection = None
_saver = None
_init_lock = Lock()
_registry_lock = Lock()
_thread_locks = {}


@lru_cache
def get_checkpointer():
    global _connection, _saver
    with _init_lock:
        if _saver is not None:
            return _saver
        try:
            path = get_settings().checkpoint_path
            path.parent.mkdir(parents=True, exist_ok=True)
            _connection = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
            _saver = SqliteSaver(_connection)
            return _saver
        except (OSError, sqlite3.Error):
            raise ServiceError("Conversation storage is unavailable. Check the API storage volume.", 503) from None


def thread_is_busy(thread_id):
    with _registry_lock:
        entry = _thread_locks.get(thread_id)
        return bool(entry and entry["lock"].locked())


@contextmanager
def thread_guard(thread_id, *, wait=False, timeout=90):
    # Keep the same lock registered until its owner AND all waiters leave.
    with _registry_lock:
        entry = _thread_locks.setdefault(thread_id, {"lock": Lock(), "users": 0})
        entry["users"] += 1
    acquired = False
    try:
        acquired = entry["lock"].acquire(timeout=timeout) if wait else entry["lock"].acquire(blocking=False)
        if not acquired:
            raise ServiceError("This chat is still processing a question. Please wait and retry.", 409)
        yield
    finally:
        if acquired:
            entry["lock"].release()
        with _registry_lock:
            entry["users"] -= 1
            if entry["users"] == 0:
                del _thread_locks[thread_id]


def shutdown_memory():
    global _connection, _saver
    with _init_lock:
        if _connection is not None:
            _connection.close()
            _connection = None
        _saver = None
        get_checkpointer.cache_clear()
