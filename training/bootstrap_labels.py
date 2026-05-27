"""
training/bootstrap_labels.py
Generate (features, label) training data for the Confidence Net by
replaying the seeded LumenX conversations through our agent pipeline.

For each seeded thread:
  1. Take customer message(s) up to (but not including) the first admin reply
  2. Generate an agent draft using Intent Router + Context Builder + LLM Draft
  3. Compute Levenshtein ratio against the actual seeded admin reply
  4. Map ratio -> label:
        sim >= 0.90 -> 1.0  (effectively identical)
        sim >= 0.70 -> 0.8  (minor edits)
        sim >= 0.40 -> 0.5  (moderate edits)
        sim >= 0.20 -> 0.2  (heavy edits)
        sim <  0.20 -> 0.0  (rejected / rewritten)
  5. Save (thread_id, draft_text, features_json, label) to the `drafts` table

Usage:
    python -m training.bootstrap_labels --limit 5     # smoke test (5 threads, ~$0.05)
    python -m training.bootstrap_labels --limit 104   # full bootstrap (~$1.15)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))

import aiosqlite
import anthropic
from Levenshtein import ratio as lev_ratio
from dotenv import load_dotenv

load_dotenv()

from agent.context_builder import ContextBuilder
from agent.intent_router import IntentRouter
from agent.llm_draft import LLMDraft
from data.database import Database
from data.models import Message, Thread
from training.feature_extractor import extract_features
from wiki.wiki_query import WikiQuery


def sim_to_label(sim: float) -> float:
    """Map Levenshtein similarity in [0,1] to training label per CLAUDE.md spec."""
    if sim >= 0.90: return 1.0
    if sim >= 0.70: return 0.8
    if sim >= 0.40: return 0.5
    if sim >= 0.20: return 0.2
    return 0.0


def split_first_admin_response(messages: list[dict]) -> tuple[list[dict], str | None]:
    """
    Given an ordered thread, return:
      - customer-side messages leading up to the first admin reply
      - the first admin reply text (the 'ground truth')
    """
    customer_msgs = []
    for m in messages:
        if m.get("role") == "admin":
            return customer_msgs, m.get("text", "")
        customer_msgs.append(m)
    return customer_msgs, None


async def load_products(db_path: str) -> dict[str, dict]:
    products: dict[str, dict] = {}
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        for row in await (await db.execute("SELECT id, raw_json FROM products_cache")).fetchall():
            products[row["id"]] = json.loads(row["raw_json"])
    return products


async def run(limit: int, dry: bool) -> None:
    api_key   = os.environ["ANTHROPIC_API_KEY"]
    db_path   = os.getenv("DB_PATH", "data/agent.db")
    wiki_path = os.getenv("WIKI_PATH", "wiki/pages")
    export    = Path("data/export_bootstrap.json")
    if not export.exists():
        print(f"ERROR: {export} not found — run scripts/bootstrap.py first")
        return

    system_prompt = (Path(__file__).parent.parent / "prompts" / "system_prompt.md").read_text(encoding="utf-8")

    print("=" * 78)
    print(f"  BOOTSTRAP TRAINING LABELS")
    print(f"  Limit:    {limit} threads  (dry={dry})")
    print(f"  Estimate: ~${0.011 * limit:.2f} in Sonnet+Haiku calls")
    print("=" * 78)

    with export.open(encoding="utf-8") as f:
        data = json.load(f)

    seeded = [
        t for t in data.get("threads", [])
        if any(m.get("role") == "admin"    for m in t.get("messages", []))
        and any(m.get("role") == "customer" for m in t.get("messages", []))
    ]
    print(f"  Eligible threads (customer + admin): {len(seeded)}")
    print()

    if dry:
        print("DRY RUN — would process the following threads:")
        for t in seeded[:limit]:
            print(f"  {t['id']:14s}  intent={t.get('intent','?'):20s}  product={t.get('product_id') or '(none)'}")
        return

    # Init agent components
    db = Database(db_path); await db.init()
    products_cache = await load_products(db_path)
    wiki_query     = WikiQuery(wiki_path)
    anthropic_client = anthropic.AsyncAnthropic(api_key=api_key)
    intent_router  = IntentRouter(anthropic_client, db)
    context_builder = ContextBuilder(wiki_query, products_cache, system_prompt)
    llm_draft      = LLMDraft(anthropic_client, db)

    label_dist = {1.0: 0, 0.8: 0, 0.5: 0, 0.2: 0, 0.0: 0}
    total_sim = 0.0
    drafts_written = 0
    skipped = 0
    t_start = time.time()

    for idx, t in enumerate(seeded[:limit], start=1):
        thread_id = t["id"]
        customer_msgs, admin_reply = split_first_admin_response(t.get("messages", []))
        if not customer_msgs or not admin_reply:
            skipped += 1
            continue

        last_customer = customer_msgs[-1].get("text", "")
        if not last_customer.strip():
            skipped += 1
            continue

        thread = Thread(
            id=thread_id,
            username=t.get("customer_username", "") or t.get("customer_display_name", ""),
            messages=[
                Message(
                    id=m.get("id", i), thread_id=thread_id,
                    role=m["role"], text=m["text"],
                    timestamp=m.get("ts", ""),
                )
                for i, m in enumerate(customer_msgs)
            ],
        )

        try:
            intent_result = await intent_router.classify(thread_id, last_customer)
            context = await context_builder.build(
                thread=thread, intent_result=intent_result,
                feedback_log=[], past_summaries="",
            )
            draft_text, in_tok, out_tok = await llm_draft.generate(thread_id, context)
        except Exception as exc:
            print(f"  [{idx:3d}] {thread_id}  ERROR: {exc}")
            skipped += 1
            continue

        sim = lev_ratio(draft_text, admin_reply)
        label = sim_to_label(sim)
        total_sim += sim
        label_dist[label] += 1

        features = extract_features(
            draft_text=draft_text,
            intent=intent_result.intent,
            thread_depth=len(thread.messages),
            wiki_hit_count=len(context.metadata.get("sections", [])),
            feedback_log_match_score=0.0,
            customer_message=last_customer,
            draft_tokens=out_tok,
        )

        # Write draft + label directly to DB (status='approved' as ground truth)
        draft_id = await db.save_draft(
            thread_id=thread_id,
            draft_text=draft_text,
            confidence_score=label,  # initial guess = label
            intent=intent_result.intent.value,
            features_json=features.model_dump(),
        )
        await db.resolve_draft(
            draft_id=draft_id,
            status="bootstrap",  # custom status so it's not shown in human-review inbox
            final_text=admin_reply,
            edit_distance=1.0 - sim,
            label=label,
        )
        drafts_written += 1
        elapsed = time.time() - t_start
        rate = drafts_written / elapsed
        eta = (min(limit, len(seeded)) - idx) / rate if rate > 0 else 0
        print(f"  [{idx:3d}/{min(limit,len(seeded)):3d}] {thread_id:14s}  intent={intent_result.intent.value:18s}  sim={sim:.2f}  label={label}  (elapsed={elapsed:5.0f}s, eta={eta:4.0f}s)")

    # ── Summary ────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print()
    print("=" * 78)
    print("  RESULTS")
    print("=" * 78)
    print(f"  Drafts written:   {drafts_written}")
    print(f"  Skipped:          {skipped}")
    print(f"  Avg similarity:   {total_sim / drafts_written if drafts_written else 0:.3f}")
    print(f"  Elapsed:          {elapsed:.0f}s ({elapsed/max(drafts_written,1):.1f}s/draft)")
    print()
    print("  Label distribution:")
    for lbl in [1.0, 0.8, 0.5, 0.2, 0.0]:
        count = label_dist[lbl]
        bar = "█" * int(count * 40 / max(sum(label_dist.values()), 1))
        print(f"    {lbl}  {count:4d}  {bar}")

    # Cost so far for this run
    async with aiosqlite.connect(db_path) as conn:
        row = await (await conn.execute(
            "SELECT SUM(cost_usd), SUM(input_tokens), SUM(output_tokens) FROM token_log "
            "WHERE timestamp > datetime('now','-1 hour')"
        )).fetchone()
    print(f"\n  Cost of this run (approx, last hour): ${row[0] or 0:.4f}")
    print(f"  Tokens: {row[1] or 0:,} in / {row[2] or 0:,} out")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5, help="Max threads to process")
    parser.add_argument("--dry", action="store_true", help="Show what would run, no API calls")
    args = parser.parse_args()
    asyncio.run(run(args.limit, args.dry))


if __name__ == "__main__":
    main()
