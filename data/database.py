"""
data/database.py
SQLite schema creation and async query helpers.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS threads (
    id               TEXT PRIMARY KEY,
    username         TEXT,
    display_name     TEXT,
    product_id       TEXT,
    intent           TEXT,
    message_count    INTEGER DEFAULT 0,
    last_customer_at TEXT,
    last_admin_at    TEXT,
    awaiting_admin   INTEGER DEFAULT 0,
    raw_json         TEXT,
    synced_at        TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE IF NOT EXISTS messages (
    id          TEXT PRIMARY KEY,
    thread_id   TEXT NOT NULL REFERENCES threads(id),
    role        TEXT NOT NULL,
    text        TEXT NOT NULL,
    timestamp   TEXT
);

CREATE TABLE IF NOT EXISTS drafts (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id        TEXT NOT NULL,
    draft_text       TEXT NOT NULL,
    confidence_score REAL,
    intent           TEXT,
    features_json    TEXT,
    status           TEXT DEFAULT 'pending',
    final_text       TEXT,
    edit_distance    REAL,
    label            REAL,
    created_at       TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    resolved_at      TEXT
);

CREATE TABLE IF NOT EXISTS token_log (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id        TEXT NOT NULL,
    timestamp        TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    step             TEXT NOT NULL,
    model            TEXT NOT NULL,
    input_tokens     INTEGER NOT NULL,
    output_tokens    INTEGER NOT NULL,
    cost_usd         REAL NOT NULL,
    context_snapshot TEXT
);

CREATE TABLE IF NOT EXISTS feedback (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    draft_id      INTEGER NOT NULL REFERENCES drafts(id),
    action        TEXT NOT NULL,
    original_text TEXT NOT NULL,
    final_text    TEXT NOT NULL,
    label         REAL NOT NULL,
    timestamp     TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE IF NOT EXISTS products_cache (
    id         TEXT PRIMARY KEY,
    name       TEXT,
    raw_json   TEXT NOT NULL,
    cached_at  TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_drafts_thread   ON drafts(thread_id);
CREATE INDEX IF NOT EXISTS idx_token_thread    ON token_log(thread_id);
CREATE INDEX IF NOT EXISTS idx_feedback_draft  ON feedback(draft_id);
CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id);
"""


# ---------------------------------------------------------------------------
# Database class
# ---------------------------------------------------------------------------

class Database:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)

    async def init(self) -> None:
        """Create tables if they don't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(SCHEMA_SQL)
            await db.commit()
        logger.info("Database initialised at %s", self.db_path)

    # ------------------------------------------------------------------
    # Token logging
    # ------------------------------------------------------------------

    async def log_tokens(
        self,
        thread_id: str,
        step: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        context_snapshot: Any = None,
    ) -> None:
        snapshot_str = json.dumps(context_snapshot) if context_snapshot else None
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT INTO token_log
                   (thread_id, step, model, input_tokens, output_tokens, cost_usd, context_snapshot)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (thread_id, step, model, input_tokens, output_tokens, cost_usd, snapshot_str),
            )
            await db.commit()

    # ------------------------------------------------------------------
    # Drafts
    # ------------------------------------------------------------------

    async def save_draft(
        self,
        thread_id: str,
        draft_text: str,
        confidence_score: float,
        intent: str,
        features_json: dict[str, Any],
    ) -> int:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """INSERT INTO drafts (thread_id, draft_text, confidence_score, intent, features_json)
                   VALUES (?, ?, ?, ?, ?)""",
                (thread_id, draft_text, confidence_score, intent, json.dumps(features_json)),
            )
            await db.commit()
            return cursor.lastrowid  # type: ignore[return-value]

    async def resolve_draft(
        self,
        draft_id: int,
        status: str,
        final_text: str,
        edit_distance: float,
        label: float,
    ) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """UPDATE drafts
                   SET status=?, final_text=?, edit_distance=?, label=?,
                       resolved_at=strftime('%Y-%m-%dT%H:%M:%fZ','now')
                   WHERE id=?""",
                (status, final_text, edit_distance, label, draft_id),
            )
            await db.commit()

    async def get_pending_drafts(self) -> list[dict[str, Any]]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM drafts WHERE status='pending' ORDER BY created_at"
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------

    async def save_feedback(
        self,
        draft_id: int,
        action: str,
        original_text: str,
        final_text: str,
        label: float,
    ) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT INTO feedback (draft_id, action, original_text, final_text, label)
                   VALUES (?, ?, ?, ?, ?)""",
                (draft_id, action, original_text, final_text, label),
            )
            await db.commit()

    async def get_labelled_dataset(self) -> list[dict[str, Any]]:
        """Return all drafts that have a label, joined with their features."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM drafts WHERE label IS NOT NULL ORDER BY created_at"
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Cost analytics
    # ------------------------------------------------------------------

    async def get_cost_summary(self, days: int = 30) -> dict[str, Any]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """SELECT
                     model,
                     SUM(input_tokens)  AS total_input,
                     SUM(output_tokens) AS total_output,
                     SUM(cost_usd)      AS total_cost,
                     COUNT(*)           AS call_count
                   FROM token_log
                   WHERE timestamp >= datetime('now', ? || ' days')
                   GROUP BY model""",
                (f"-{days}",),
            )
            rows = await cursor.fetchall()
            return {"period_days": days, "by_model": [dict(r) for r in rows]}
