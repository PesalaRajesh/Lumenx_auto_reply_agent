"""
training/feature_extractor.py
Extracts ConfidenceFeatures from a draft + its context metadata.
"""
from __future__ import annotations

import re

from data.models import ConfidenceFeatures, IntentLabel

# Intent → integer mapping (deterministic)
_INTENT_MAP = {
    IntentLabel.GREETING:           0,
    IntentLabel.GENERIC_CHAT:       1,
    IntentLabel.PRODUCT_INFO:       2,
    IntentLabel.PRICING:            3,
    IntentLabel.REFUND_POLICY:      4,
    IntentLabel.TECHNICAL_SUPPORT:  5,
    IntentLabel.COMPLAINT:          6,
    IntentLabel.UNKNOWN:            7,
}

_PRICING_KEYWORDS  = re.compile(r"\$|\bprice\b|\bpricing\b|\bplan\b|\bcost\b|\bper month\b|\bsubscription\b", re.I)
_REFUND_KEYWORDS   = re.compile(r"\brefund\b|\bcancel\b|\bcancellation\b|\bmoney.?back\b", re.I)
_SENTIMENT_NEG     = re.compile(r"\bfrustrat\b|\bangr\b|\bdisappoint\b|\bterrible\b|\bawful\b|\bbroke\b", re.I)
_SENTIMENT_POS     = re.compile(r"\bthank\b|\bgreat\b|\bamazing\b|\bwonderful\b|\bappreciat\b", re.I)


def extract_features(
    *,
    draft_text: str,
    intent: IntentLabel,
    thread_depth: int,
    wiki_hit_count: int,
    feedback_log_match_score: float,
    customer_message: str,
    expected_reply_tokens: int = 200,
    draft_tokens: int | None = None,
) -> ConfidenceFeatures:
    """Extract normalised features from a draft + context."""

    words = draft_text.split()
    word_count = len(words)

    # Normalise word count: typical range 50–300 words → map to [0, 1]
    reply_length_norm = min(word_count / 300, 1.0)

    # Thread depth: typical range 0–20 → normalise
    thread_depth_norm = min(thread_depth / 20, 1.0)

    # Wiki hit count: 0–5 typically
    wiki_norm = min(wiki_hit_count / 5, 1.0)

    # Draft length ratio
    if draft_tokens is None:
        draft_tokens = max(1, word_count * 4 // 3)  # rough tokens estimate
    length_ratio = min(draft_tokens / max(expected_reply_tokens, 1), 2.0) / 2.0

    # Sentiment from customer message
    if _SENTIMENT_NEG.search(customer_message):
        sentiment = -1.0
    elif _SENTIMENT_POS.search(customer_message):
        sentiment = 1.0
    else:
        sentiment = 0.0

    return ConfidenceFeatures(
        intent_id=_INTENT_MAP.get(intent, 7),
        reply_length=reply_length_norm,
        contains_pricing=1.0 if _PRICING_KEYWORDS.search(draft_text) else 0.0,
        contains_refund=1.0 if _REFUND_KEYWORDS.search(draft_text) else 0.0,
        product_mentioned=1.0 if any(
            p in draft_text.lower()
            for p in ["emailpilot", "invoiceflow", "taskgrid"]
        ) else 0.0,
        thread_depth=thread_depth_norm,
        wiki_hit_count=wiki_norm,
        feedback_log_match_score=max(0.0, min(feedback_log_match_score, 1.0)),
        draft_length_ratio=length_ratio,
        customer_sentiment=sentiment,
    )
