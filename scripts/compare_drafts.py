"""Show generated draft vs seeded admin reply side by side, for inspection."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

conn = sqlite3.connect("data/agent.db")
conn.row_factory = sqlite3.Row
rows = conn.execute(
    "SELECT thread_id, draft_text, final_text, edit_distance, label, intent FROM drafts "
    "WHERE status='bootstrap' ORDER BY id DESC LIMIT 3"
).fetchall()

for r in rows:
    print("=" * 78)
    print(f"Thread: {r['thread_id']} | intent: {r['intent']} | sim={1-r['edit_distance']:.2f} | label={r['label']}")
    print("=" * 78)
    print("\n--- AGENT DRAFT ---")
    print(r["draft_text"])
    print(f"\n  [length: {len(r['draft_text'])} chars / {len(r['draft_text'].split())} words]")
    print("\n--- SEEDED ADMIN REPLY (ground truth) ---")
    print(r["final_text"])
    print(f"\n  [length: {len(r['final_text'])} chars / {len(r['final_text'].split())} words]")
    print()

conn.close()
