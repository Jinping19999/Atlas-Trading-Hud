"""
Atlas Trading HUD — SQLite persistence layer.

Simple key-value store that mirrors the browser's localStorage model.
Keys: signal_history, monitor, archive

Storage location:
  - DATA_DIR env var (set to Railway volume mount, e.g. /data)
  - Falls back to app directory for local development
"""

import os
import json
import sqlite3
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger("atlas-db")

# ── Database path ───────────────────────────────────────────────────────────
DATA_DIR = os.environ.get("DATA_DIR", os.path.dirname(__file__) or ".")
DB_PATH = os.path.join(DATA_DIR, "atlas.db")


def _get_conn() -> sqlite3.Connection:
    """Get a SQLite connection with WAL mode for concurrent reads."""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def init_db():
    """Create tables if they don't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = _get_conn()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS kv_store (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.commit()
        log.info(f"Database initialized at {DB_PATH}")
    finally:
        conn.close()


# ── Generic KV operations ──────────────────────────────────────────────────

def kv_get(key: str) -> Optional[dict | list]:
    """Read a JSON value by key. Returns None if not found."""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT value FROM kv_store WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])
    except Exception as e:
        log.warning(f"kv_get({key}) failed: {e}")
        return None
    finally:
        conn.close()


def kv_set(key: str, value) -> bool:
    """Write a JSON value by key. Returns True on success."""
    conn = _get_conn()
    try:
        json_str = json.dumps(value, default=str)
        conn.execute(
            """INSERT INTO kv_store (key, value, updated_at)
               VALUES (?, ?, datetime('now'))
               ON CONFLICT(key) DO UPDATE SET
                 value = excluded.value,
                 updated_at = datetime('now')
            """,
            (key, json_str),
        )
        conn.commit()
        return True
    except Exception as e:
        log.warning(f"kv_set({key}) failed: {e}")
        return False
    finally:
        conn.close()


def kv_delete(key: str) -> bool:
    """Delete a key. Returns True on success."""
    conn = _get_conn()
    try:
        conn.execute("DELETE FROM kv_store WHERE key = ?", (key,))
        conn.commit()
        return True
    except Exception as e:
        log.warning(f"kv_delete({key}) failed: {e}")
        return False
    finally:
        conn.close()


# ── Typed helpers ───────────────────────────────────────────────────────────

# Signal history: dict of ticker → {firstSeen, lastSeen, daysSeen, lastData}
SIGNAL_HISTORY_KEY = "signal_history"

def get_signal_history() -> dict:
    return kv_get(SIGNAL_HISTORY_KEY) or {}

def save_signal_history(history: dict) -> bool:
    return kv_set(SIGNAL_HISTORY_KEY, history)


# Monitor: list of position dicts (active positions being tracked)
MONITOR_KEY = "monitor"

def get_monitor() -> list:
    return kv_get(MONITOR_KEY) or []

def save_monitor(positions: list) -> bool:
    return kv_set(MONITOR_KEY, positions)


# Archive: list of completed/expired monitor positions
ARCHIVE_KEY = "archive"

def get_archive() -> list:
    return kv_get(ARCHIVE_KEY) or []

def save_archive(archive: list) -> bool:
    # Keep last 200 entries max
    if len(archive) > 200:
        archive = archive[-200:]
    return kv_set(ARCHIVE_KEY, archive)


# ── Bulk read (single round-trip for page load sync) ────────────────────────

def get_all_state() -> dict:
    """Read all persisted state in one call (for frontend sync on load)."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT key, value FROM kv_store WHERE key IN (?, ?, ?)",
            (SIGNAL_HISTORY_KEY, MONITOR_KEY, ARCHIVE_KEY),
        ).fetchall()
        result = {
            "signal_history": {},
            "monitor": [],
            "archive": [],
        }
        for key, value in rows:
            try:
                result[key] = json.loads(value)
            except Exception:
                pass
        return result
    except Exception as e:
        log.warning(f"get_all_state failed: {e}")
        return {"signal_history": {}, "monitor": [], "archive": []}
    finally:
        conn.close()
