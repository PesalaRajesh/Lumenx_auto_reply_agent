"""
scripts/test_pipeline.py
One-shot end-to-end pipeline runner for Phases 3+4+5.

Picks ONE unread customer thread from LumenX, runs:
  Intent Router (Haiku) -> Context Builder -> LLM Draft (Sonnet) -> save to DB
Does NOT send to LumenX. Use --send to actually post the reply.

Usage:
    python scripts/test_pipeline.py                 # dry run, save draft only
    python scripts/test_pipeline.py --thread-id X   # target a specific thread
    python scripts/test_pipeline.py --send          # actually send the draft
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# UTF-8 stdout on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))

import aiosqlite
import anthropic
from dotenv import load_dotenv

load_dotenv()

from agent.confidence_net import ConfidenceNet
from agent.context_builder import ContextBuilder
from agent.intent_router import IntentRouter
from agent.llm_draft import LLMDraft
from data.database import Database
from data.lumenx_client import LumenXClient
from data.models import Message, Thread
from training.feature_extractor import extract_features
from wiki.wiki_query import WikiQuery


def banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def section(title: str) -> None:
    print(f"\n--- {title} ---")


async def load_products_cache(db_path: str) -> dict[str, dict]:
    products: dict[str, dict] = {}
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT id, raw_json FROM products_cache")
        for row in await cursor.fetchall():
            products[row["id"]] = json.loads(row["raw_json"])
    return products


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thread-id", help="Specific thread ID to process (default: first awaiting)")
    parser.add_argument("--send", action="store_true", help="Actually POST the reply to LumenX")
    args = parser.parse_args()

    base_url    = os.environ["LUMENX_BASE_URL"]
    admin_token = os.environ["LUMENX_ADMIN_TOKEN"]
    api_key     = os.environ["ANTHROPIC_API_KEY"]
    db_path     = os.getenv("DB_PATH", "data/agent.db")
    wiki_path   = os.getenv("WIKI_PATH", "wiki/pages")
    system_prompt = (Path(__file__).parent.parent / "prompts" / "system_prompt.md").read_text(encoding="utf-8")

    banner("PHASE 3+4+5 END-TO-END SMOKE TEST")
    print(f"  Send: {'YES — will post to LumenX!' if args.send else 'NO (dry run)'}")
    print(f"  DB:   {db_path}")
    print(f"  Wiki: {wiki_path}")

    db = Database(db_path)
    await db.init()
    products_cache = await load_products_cache(db_path)
    wiki_query     = WikiQuery(wiki_path)
    anthropic_client = anthropic.AsyncAnthropic(api_key=api_key)
    confidence_net = ConfidenceNet()

    print(f"  Products cached: {len(products_cache)}")
    print(f"  MLP trained:     {confidence_net.is_trained} (untrained -> fallback score 0.5)")

    intent_router  = IntentRouter(anthropic_client, db)
    context_builder = ContextBuilder(wiki_query, products_cache, system_prompt)
    llm_draft      = LLMDraft(anthropic_client, db)

    async with LumenXClient(base_url, admin_token) as lumenx:
        # ── Pick a thread ─────────────────────────────────────
        if args.thread_id:
            thread_id = args.thread_id
        else:
            inbox = await lumenx.get_inbox()
            entries = inbox.get("entries", [])
            if not entries:
                print("\nNo unread threads. Try posting a customer message via the LumenX chat UI.")
                return
            # First entry that has a thread object
            thread_id = entries[0]["thread"]["id"] if "thread" in entries[0] else entries[0].get("id")

        banner(f"PROCESSING THREAD: {thread_id}")

        # ── Fetch thread ─────────────────────────────────────
        raw = await lumenx.get_thread(thread_id)
        # LumenX returns: { "thread": { id, customer_username, messages: [...] } }
        t = raw.get("thread", raw)
        thread = Thread(
            id=t["id"],
            username=t.get("customer_username", "") or t.get("customer_display_name", ""),
            messages=[
                Message(
                    id=m.get("id", i),
                    thread_id=thread_id,
                    role=m["role"],
                    text=m["text"],
                    timestamp=m.get("ts") or m.get("timestamp", ""),
                )
                for i, m in enumerate(t.get("messages", []))
            ],
        )
        last_msg = next((m.text for m in reversed(thread.messages) if m.role == "customer"), None)
        if not last_msg:
            print("  Thread has no customer message — exiting.")
            return

        section("Thread context")
        print(f"  Customer: {thread.username}")
        print(f"  Messages: {len(thread.messages)}")
        print(f"  Last customer message: {last_msg[:200]}{'...' if len(last_msg) > 200 else ''}")

        # ── PHASE 3: Intent Router ────────────────────────────
        section("Phase 3 — Intent Router (Haiku)")
        intent_result = await intent_router.classify(thread_id, last_msg)
        print(f"  Intent:     {intent_result.intent.value}")
        print(f"  Product:    {intent_result.product_id or '(none)'}")
        print(f"  Confidence: {intent_result.confidence:.2f}")
        print(f"  Reasoning:  {intent_result.reasoning}")

        # ── PHASE 4: Context Builder ──────────────────────────
        section("Phase 4 — Context Builder")
        context = await context_builder.build(
            thread=thread,
            intent_result=intent_result,
            feedback_log=[],
            past_summaries="",
        )
        print(f"  Sections included: {context.metadata.get('sections', [])}")
        print(f"  User content chars: {len(context.user_content)}")

        # ── PHASE 5: LLM Draft (Sonnet) ───────────────────────
        section("Phase 5 — LLM Draft (Sonnet)")
        draft_text, in_tok, out_tok = await llm_draft.generate(thread_id, context)
        print(f"  Tokens: {in_tok} in / {out_tok} out")
        print(f"\n  DRAFT:\n  " + "─" * 60)
        for line in draft_text.split("\n"):
            print(f"    {line}")
        print("  " + "─" * 60)

        # ── Confidence scoring ────────────────────────────────
        section("Confidence Net scoring")
        features = extract_features(
            draft_text=draft_text,
            intent=intent_result.intent,
            thread_depth=len(thread.messages),
            wiki_hit_count=len(context.metadata.get("sections", [])),
            feedback_log_match_score=0.0,
            customer_message=last_msg,
            draft_tokens=out_tok,
        )
        score = confidence_net.predict(features)
        print(f"  Confidence score: {score:.3f}  ({'TRAINED' if confidence_net.is_trained else 'fallback (untrained)'})")
        print(f"  Features: intent_id={features.intent_id}, len={features.reply_length:.2f}, " +
              f"pricing={features.contains_pricing}, refund={features.contains_refund}, " +
              f"sentiment={features.customer_sentiment}")

        # ── Save to DB ────────────────────────────────────────
        draft_id = await db.save_draft(
            thread_id=thread_id,
            draft_text=draft_text,
            confidence_score=score,
            intent=intent_result.intent.value,
            features_json=features.model_dump(),
        )
        print(f"\n  Saved as draft #{draft_id} in DB (status=pending)")

        # ── Optionally send ───────────────────────────────────
        if args.send:
            section("Sending reply to LumenX")
            await lumenx.send_reply(thread_id, draft_text, draft_source="agent", confidence=score)
            await db.resolve_draft(draft_id, status="auto_sent", final_text=draft_text, edit_distance=0.0, label=1.0)
            print("  Sent.")
        else:
            print("\n  (Dry run — draft NOT sent. Review it at http://127.0.0.1:8080/inbox)")

    banner("COMPLETE")


if __name__ == "__main__":
    asyncio.run(main())
