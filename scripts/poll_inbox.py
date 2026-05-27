"""
scripts/poll_inbox.py
Inbox polling daemon — checks for new customer messages and triggers
the agent pipeline (intent → context → draft → confidence → route).

Usage:
    python scripts/poll_inbox.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from pathlib import Path

import anthropic
import aiosqlite
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()
console = Console()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from agent.confidence_net import ConfidenceNet
from agent.context_builder import ContextBuilder
from agent.intent_router import IntentRouter
from agent.llm_draft import LLMDraft
from data.database import Database
from data.lumenx_client import LumenXClient
from data.models import IntentLabel, Thread, Message
from training.feature_extractor import extract_features
from wiki.wiki_query import WikiQuery


async def load_products_cache(db_path: str) -> dict[str, dict]:
    products = {}
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT id, raw_json FROM products_cache")
        rows = await cursor.fetchall()
        for row in rows:
            products[row["id"]] = json.loads(row["raw_json"])
    return products


async def process_thread(
    thread_id: str,
    lumenx: LumenXClient,
    intent_router: IntentRouter,
    context_builder: ContextBuilder,
    llm_draft: LLMDraft,
    confidence_net: ConfidenceNet,
    db: Database,
    confidence_threshold: float,
    review_sample_rate: float,
) -> None:
    """Run the full agent pipeline for one thread."""

    # Fetch full thread
    raw_thread = await lumenx.get_thread(thread_id)
    # LumenX returns: { "thread": { id, customer_username, messages: [...] } }
    t = raw_thread.get("thread", raw_thread)
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

    # Get last customer message
    last_msg = next(
        (m.text for m in reversed(thread.messages) if m.role == "customer"), None
    )
    if not last_msg:
        logger.warning("Thread %s has no customer message — skipping", thread_id)
        return

    # 1. Intent routing
    intent_result = await intent_router.classify(thread_id, last_msg)

    # If it's a simple greeting/chat with no product relevance, use a quick path
    if intent_result.intent in (IntentLabel.GREETING, IntentLabel.GENERIC_CHAT):
        logger.info("Thread %s: simple intent %s — fast path", thread_id, intent_result.intent)

    # 2. Context building
    # (simplified: no past summaries or feedback log in this skeleton)
    context = await context_builder.build(
        thread=thread,
        intent_result=intent_result,
        feedback_log=[],
        past_summaries="",
    )

    # 3. Draft generation
    draft_text, in_tok, out_tok = await llm_draft.generate(thread_id, context)

    # 4. Feature extraction + confidence scoring
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

    # 5. Save draft to DB
    draft_id = await db.save_draft(
        thread_id=thread_id,
        draft_text=draft_text,
        confidence_score=score,
        intent=intent_result.intent,
        features_json=features.model_dump(),
    )

    # 6. Route: auto-send or human review
    is_trained = confidence_net.is_trained
    force_review = random.random() < review_sample_rate  # active learning sample

    if is_trained and score >= confidence_threshold and not force_review:
        # AUTO-SEND
        logger.info(
            "Thread %s: AUTO-SEND (score=%.3f >= threshold=%.3f)",
            thread_id, score, confidence_threshold,
        )
        await lumenx.send_reply(
            thread_id=thread_id,
            text=draft_text,
            draft_source="agent",
            confidence=score,
        )
        await db.resolve_draft(
            draft_id=draft_id,
            status="auto_sent",
            final_text=draft_text,
            edit_distance=0.0,
            label=1.0,
        )
    else:
        # HUMAN REVIEW — draft saved to DB; dashboard picks it up
        reason = "low confidence" if score < confidence_threshold else (
            "not yet trained" if not is_trained else "random sample"
        )
        logger.info(
            "Thread %s: HUMAN REVIEW (score=%.3f, reason=%s)",
            thread_id, score, reason,
        )


async def main() -> None:
    base_url           = os.environ["LUMENX_BASE_URL"]
    admin_token        = os.environ["LUMENX_ADMIN_TOKEN"]
    db_path            = os.getenv("DB_PATH", "data/agent.db")
    wiki_path          = os.getenv("WIKI_PATH", "wiki/pages")
    api_key            = os.environ["ANTHROPIC_API_KEY"]
    poll_interval      = int(os.getenv("POLL_INTERVAL_SECONDS", "10"))
    threshold          = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))
    review_sample_rate = float(os.getenv("REVIEW_SAMPLE_RATE", "0.10"))

    system_prompt_path = Path(__file__).parent.parent / "prompts" / "system_prompt.md"
    system_prompt      = system_prompt_path.read_text(encoding="utf-8")

    db              = Database(db_path)
    await db.init()

    products_cache  = await load_products_cache(db_path)
    wiki_query      = WikiQuery(wiki_path)
    anthropic_client = anthropic.AsyncAnthropic(api_key=api_key)
    confidence_net  = ConfidenceNet()

    intent_router   = IntentRouter(anthropic_client, db)
    context_builder = ContextBuilder(wiki_query, products_cache, system_prompt)
    llm_draft_gen   = LLMDraft(anthropic_client, db)

    console.rule("[bold green]LumenX Inbox Poller Started")
    console.print(f"  Polling every {poll_interval}s | threshold={threshold} | sample_rate={review_sample_rate}")

    server_time: str | None = None

    async with LumenXClient(base_url, admin_token) as lumenx:
        while True:
            try:
                inbox = await lumenx.get_inbox(since=server_time)
                server_time = inbox.get("server_time")
                entries     = inbox.get("entries", [])

                for entry in entries:
                    thread_id = entry.get("thread", {}).get("id") or entry.get("id")
                    if not thread_id:
                        continue
                    try:
                        await process_thread(
                            thread_id=thread_id,
                            lumenx=lumenx,
                            intent_router=intent_router,
                            context_builder=context_builder,
                            llm_draft=llm_draft_gen,
                            confidence_net=confidence_net,
                            db=db,
                            confidence_threshold=threshold,
                            review_sample_rate=review_sample_rate,
                        )
                    except Exception as exc:
                        logger.error("Error processing thread %s: %s", thread_id, exc, exc_info=True)

            except Exception as exc:
                logger.error("Inbox poll error: %s", exc, exc_info=True)

            await asyncio.sleep(poll_interval)


if __name__ == "__main__":
    asyncio.run(main())
