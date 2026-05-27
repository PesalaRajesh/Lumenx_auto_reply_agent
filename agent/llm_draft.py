"""
agent/llm_draft.py
Generates reply drafts using Claude Sonnet.
Uses prompt caching for the system prompt + product knowledge.
"""
from __future__ import annotations

import logging

import anthropic

from agent.context_builder import BuiltContext
from data.database import Database

logger = logging.getLogger(__name__)

_MODEL = "claude-sonnet-4-6"
_SONNET_INPUT_COST  = 3.00   # per million tokens
_SONNET_OUTPUT_COST = 15.00  # per million tokens


class LLMDraft:
    """Generates a reply draft using Claude Sonnet with prompt caching."""

    def __init__(self, client: anthropic.AsyncAnthropic, db: Database) -> None:
        self._client = client
        self._db = db

    async def generate(self, thread_id: str, context: BuiltContext) -> tuple[str, int, int]:
        """
        Generate a reply draft.
        Returns: (draft_text, input_tokens, output_tokens)
        """
        # Use prompt caching for the system prompt (it's static and > 1024 tokens)
        response = await self._client.messages.create(
            model=_MODEL,
            max_tokens=512,
            system=[
                {
                    "type": "text",
                    "text": context.system_prompt,
                    "cache_control": {"type": "ephemeral"},  # Cache the system prompt
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": context.user_content,
                }
            ],
        )

        input_tokens  = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost = (
            input_tokens  * _SONNET_INPUT_COST  +
            output_tokens * _SONNET_OUTPUT_COST
        ) / 1_000_000

        await self._db.log_tokens(
            thread_id=thread_id,
            step="llm_draft",
            model=_MODEL,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            context_snapshot={
                "sections": context.metadata.get("sections", []),
                "user_content_length": len(context.user_content),
            },
        )

        draft_text = response.content[0].text.strip()
        logger.info(
            "Draft generated for thread %s (%d in / %d out tokens, $%.5f)",
            thread_id, input_tokens, output_tokens, cost,
        )
        return draft_text, input_tokens, output_tokens
