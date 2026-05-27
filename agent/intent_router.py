"""
agent/intent_router.py
Classifies incoming customer messages into intent categories.
Uses Claude Haiku for cost-effective classification.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import anthropic

from data.database import Database
from data.models import IntentLabel, IntentResult

logger = logging.getLogger(__name__)

# Cost constants (per million tokens)
_HAIKU_INPUT_COST  = 0.80
_HAIKU_OUTPUT_COST = 4.00
_MODEL = "claude-haiku-4-5"

_PROMPT_TEMPLATE = (Path(__file__).parent.parent / "prompts" / "intent_prompt.md").read_text()


class IntentRouter:
    """Classifies customer messages into structured intent categories."""

    def __init__(self, client: anthropic.AsyncAnthropic, db: Database) -> None:
        self._client = client
        self._db = db

    async def classify(self, thread_id: str, message: str) -> IntentResult:
        """
        Classify `message` and return an IntentResult.
        Logs token usage to the database.
        """
        prompt = _PROMPT_TEMPLATE.replace("{{message}}", message)

        response = await self._client.messages.create(
            model=_MODEL,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )

        input_tokens  = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost = (input_tokens * _HAIKU_INPUT_COST + output_tokens * _HAIKU_OUTPUT_COST) / 1_000_000

        await self._db.log_tokens(
            thread_id=thread_id,
            step="intent_router",
            model=_MODEL,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )

        raw = response.content[0].text.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        try:
            data = json.loads(raw)
            result = IntentResult(
                intent=IntentLabel(data.get("intent", "unknown")),
                product_id=data.get("product_id"),
                confidence=float(data.get("confidence", 1.0)),
                reasoning=data.get("reasoning"),
            )
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Intent parse failed (%s) — defaulting to unknown", exc)
            result = IntentResult(intent=IntentLabel.UNKNOWN)

        logger.info(
            "Intent[%s] = %s (product=%s, conf=%.2f)",
            thread_id, result.intent, result.product_id, result.confidence,
        )
        return result
