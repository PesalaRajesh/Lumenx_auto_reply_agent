"""
training/augment_labels.py
Add synthetic spread to the bootstrap training set so the MLP has something to learn.

For each existing bootstrap draft (label ~0.5), this script adds:
  * a row using the SEEDED ADMIN REPLY as the draft text -> label = 1.0
    (by definition the admin reply was 'sent unchanged' — that's the 1.0 case)
  * an optionally truncated, low-information version -> label = 0.0
    (proxy for a 'rejected / heavily rewritten' draft)

Result: ~3x more training examples with full label spread {0.0, 0.5, 1.0}.

Usage:
    python -m training.augment_labels
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))

import aiosqlite
from dotenv import load_dotenv

load_dotenv()

from data.database import Database
from data.models import IntentLabel
from training.feature_extractor import extract_features


async def run() -> None:
    db_path = os.getenv("DB_PATH", "data/agent.db")
    db = Database(db_path)

    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row
        cursor = await conn.execute(
            "SELECT * FROM drafts WHERE status='bootstrap' AND label IS NOT NULL"
        )
        bootstrap_rows = [dict(r) for r in await cursor.fetchall()]

    print(f"Found {len(bootstrap_rows)} bootstrap drafts to augment.")

    added_perfect = 0
    added_bad     = 0

    for row in bootstrap_rows:
        # Load the original features so we can preserve intent/context fields
        orig_features = json.loads(row["features_json"])
        intent = IntentLabel(row["intent"])
        admin_reply = row["final_text"] or ""

        # ── (1) PERFECT example: the seeded admin reply with label=1.0 ────
        if admin_reply.strip():
            features_perfect = extract_features(
                draft_text=admin_reply,
                intent=intent,
                thread_depth=int(orig_features.get("thread_depth", 0.05) * 20),
                wiki_hit_count=int(orig_features.get("wiki_hit_count", 0.2) * 5),
                feedback_log_match_score=orig_features.get("feedback_log_match_score", 0.0),
                customer_message="",  # not used in feature extraction
                draft_tokens=len(admin_reply.split()) * 4 // 3,
            )
            draft_id = await db.save_draft(
                thread_id=row["thread_id"] + "::perfect",
                draft_text=admin_reply,
                confidence_score=1.0,
                intent=intent.value,
                features_json=features_perfect.model_dump(),
            )
            await db.resolve_draft(
                draft_id=draft_id, status="augment_perfect",
                final_text=admin_reply, edit_distance=0.0, label=1.0,
            )
            added_perfect += 1

        # ── (2) BAD example: truncated / abrupt version with label=0.0 ─────
        original_draft = row["draft_text"]
        words = original_draft.split()
        if len(words) > 10:
            # Use only the first 20% of words — feels like a cut-off, incomplete reply
            cut = max(5, len(words) // 5)
            bad_text = " ".join(words[:cut]) + "..."
            features_bad = extract_features(
                draft_text=bad_text,
                intent=intent,
                thread_depth=int(orig_features.get("thread_depth", 0.05) * 20),
                wiki_hit_count=int(orig_features.get("wiki_hit_count", 0.2) * 5),
                feedback_log_match_score=orig_features.get("feedback_log_match_score", 0.0),
                customer_message="",
                draft_tokens=cut * 4 // 3,
            )
            draft_id = await db.save_draft(
                thread_id=row["thread_id"] + "::bad",
                draft_text=bad_text,
                confidence_score=0.0,
                intent=intent.value,
                features_json=features_bad.model_dump(),
            )
            await db.resolve_draft(
                draft_id=draft_id, status="augment_bad",
                final_text="", edit_distance=1.0, label=0.0,
            )
            added_bad += 1

    print(f"Added {added_perfect} 'perfect' (label=1.0) examples")
    print(f"Added {added_bad} 'bad'     (label=0.0) examples")

    async with aiosqlite.connect(db_path) as conn:
        cursor = await conn.execute(
            "SELECT label, COUNT(*) FROM drafts WHERE label IS NOT NULL GROUP BY label ORDER BY label"
        )
        rows = await cursor.fetchall()
    print("\nFinal label distribution in DB:")
    for lbl, cnt in rows:
        print(f"  label={lbl}  count={cnt}")


if __name__ == "__main__":
    asyncio.run(run())
