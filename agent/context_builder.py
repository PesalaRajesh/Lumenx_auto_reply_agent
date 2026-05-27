"""
agent/context_builder.py
Assembles the token-budgeted context window for the LLM Draft step.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from data.models import IntentLabel, IntentResult, Message, Thread
from wiki.wiki_query import WikiQuery

logger = logging.getLogger(__name__)

# Token budget (approximate — Haiku tokenisation used for estimation)
TOKEN_BUDGET = {
    "system_prompt":   800,
    "current_thread":  3000,
    "product_json":    2000,
    "wiki_pages":      2500,
    "past_summaries":  1500,
    "feedback_log":    1200,
}

# Rough chars-per-token for budget enforcement
_CHARS_PER_TOKEN = 4


def _truncate_to_budget(text: str, token_budget: int) -> str:
    char_budget = token_budget * _CHARS_PER_TOKEN
    if len(text) <= char_budget:
        return text
    return text[:char_budget] + "\n...[truncated]"


@dataclass
class BuiltContext:
    system_prompt: str
    user_content: str
    metadata: dict[str, Any]  # debug info for dashboard


class ContextBuilder:
    def __init__(
        self,
        wiki_query: WikiQuery,
        products_cache: dict[str, dict[str, Any]],
        system_prompt: str,
    ) -> None:
        self._wiki = wiki_query
        self._products = products_cache
        self._system_prompt = system_prompt

    async def build(
        self,
        thread: Thread,
        intent_result: IntentResult,
        feedback_log: list[dict[str, Any]],
        past_summaries: str,
    ) -> BuiltContext:
        """Assemble context and return BuiltContext with both system + user messages."""

        sections: list[str] = []
        metadata: dict[str, Any] = {"intent": intent_result.intent, "sections": []}

        # ── 0. Customer identity (lets the agent greet by name) ───────────
        first_name = self._first_name(thread.username)
        if first_name:
            sections.append(f"## Customer\nFirst name: **{first_name}**  (use this name in your greeting)")
            metadata["sections"].append(f"customer:{first_name}")

        # ── 1. Current thread history ─────────────────────────────────────
        thread_text = self._format_thread(thread)
        thread_text = _truncate_to_budget(thread_text, TOKEN_BUDGET["current_thread"])
        sections.append(f"## Current Conversation Thread\n{thread_text}")
        metadata["sections"].append("current_thread")

        # ── 2. Product JSON (if product intent) ──────────────────────────
        product_json_text = ""
        if intent_result.product_id and intent_result.product_id in self._products:
            product_data = self._products[intent_result.product_id]
            product_json_text = json.dumps(product_data, indent=2)
            product_json_text = _truncate_to_budget(product_json_text, TOKEN_BUDGET["product_json"])
            sections.append(f"## Product Information\n```json\n{product_json_text}\n```")
            metadata["sections"].append(f"product:{intent_result.product_id}")
        elif intent_result.intent in (
            IntentLabel.PRICING, IntentLabel.REFUND_POLICY
        ):
            # Load all products for policy/pricing intents
            all_products = json.dumps(list(self._products.values()), indent=2)
            all_products = _truncate_to_budget(all_products, TOKEN_BUDGET["product_json"])
            sections.append(f"## All Products & Policies\n```json\n{all_products}\n```")
            metadata["sections"].append("all_products")

        # ── 3. Wiki pages (BM25 retrieval) ───────────────────────────────
        last_message = self._last_customer_message(thread)
        if last_message:
            wiki_pages = await self._wiki.query(last_message, top_k=3)
            if wiki_pages:
                wiki_text = "\n\n---\n\n".join(
                    f"**Wiki: {p['title']}**\n{p['content']}" for p in wiki_pages
                )
                wiki_text = _truncate_to_budget(wiki_text, TOKEN_BUDGET["wiki_pages"])
                sections.append(f"## Knowledge Base\n{wiki_text}")
                metadata["sections"].append(f"wiki:{len(wiki_pages)} pages")

        # ── 4. Past conversation summaries ────────────────────────────────
        if past_summaries:
            summaries_text = _truncate_to_budget(past_summaries, TOKEN_BUDGET["past_summaries"])
            sections.append(f"## Summary of Past Conversations\n{summaries_text}")
            metadata["sections"].append("past_summaries")

        # ── 5. Feedback log (similar past Q&A) ───────────────────────────
        if feedback_log:
            log_text = self._format_feedback_log(feedback_log)
            log_text = _truncate_to_budget(log_text, TOKEN_BUDGET["feedback_log"])
            sections.append(f"## Relevant Past Resolved Conversations\n{log_text}")
            metadata["sections"].append(f"feedback_log:{len(feedback_log)} items")

        user_content = "\n\n".join(sections)
        user_content += "\n\n---\n\nPlease draft a reply to the latest customer message above."

        return BuiltContext(
            system_prompt=self._system_prompt,
            user_content=user_content,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _first_name(username: str | None) -> str:
        """Extract a first name from `username` ('Lucas_Tanaka' -> 'Lucas')."""
        if not username:
            return ""
        # Split on underscore, space, or dot; take the first token
        import re as _re
        parts = _re.split(r"[_\s.]+", username.strip())
        if not parts or not parts[0]:
            return ""
        first = parts[0]
        return first.title() if first.islower() else first

    @staticmethod
    def _format_thread(thread: Thread) -> str:
        lines: list[str] = []
        for msg in thread.messages[-20:]:  # last 20 messages max
            role_label = "Customer" if msg.role == "customer" else "Support Agent"
            lines.append(f"[{role_label}]: {msg.text}")
        return "\n".join(lines)

    @staticmethod
    def _last_customer_message(thread: Thread) -> str | None:
        for msg in reversed(thread.messages):
            if msg.role == "customer":
                return msg.text
        return None

    @staticmethod
    def _format_feedback_log(items: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for item in items[:5]:  # top 5
            lines.append(
                f"Q: {item.get('question', '')}\n"
                f"A: {item.get('answer', '')}\n"
                f"(Confidence: {item.get('label', 0):.2f})"
            )
        return "\n\n".join(lines)
