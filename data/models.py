"""
data/models.py
Pydantic v2 data models for the LumenX Auto-Reply Agent.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class IntentLabel(str, Enum):
    GREETING = "greeting"
    GENERIC_CHAT = "generic_chat"
    PRODUCT_INFO = "product_info"
    PRICING = "pricing"
    REFUND_POLICY = "refund_policy"
    TECHNICAL_SUPPORT = "technical_support"
    COMPLAINT = "complaint"
    UNKNOWN = "unknown"


class DraftStatus(str, Enum):
    PENDING = "pending"
    AUTO_SENT = "auto_sent"
    APPROVED = "approved"
    EDITED = "edited"
    REJECTED = "rejected"


class FeedbackAction(str, Enum):
    APPROVE = "approve"
    EDIT = "edit"
    REJECT = "reject"


# ---------------------------------------------------------------------------
# LumenX API models (mirrors of LumenX REST responses)
# ---------------------------------------------------------------------------

class Message(BaseModel):
    id: str | int
    thread_id: str
    role: str  # "customer" | "admin"
    text: str
    timestamp: datetime | str


class Thread(BaseModel):
    id: str
    username: str
    display_name: str | None = None
    product_id: str | None = None
    intent: str | None = None
    message_count: int = 0
    last_customer_at: datetime | str | None = None
    last_admin_at: datetime | str | None = None
    awaiting_admin: bool = False
    messages: list[Message] = Field(default_factory=list)


class Product(BaseModel):
    id: str
    name: str
    description: str | None = None
    pricing_tiers: Any = None
    features: list[str] = Field(default_factory=list)
    refund_policy: str | None = None
    cancellation_policy: str | None = None
    integrations: list[str] = Field(default_factory=list)
    target_audience: str | None = None
    support_sla: str | None = None
    free_trial: str | None = None


# ---------------------------------------------------------------------------
# Agent internal models
# ---------------------------------------------------------------------------

class IntentResult(BaseModel):
    intent: IntentLabel
    product_id: str | None = None
    confidence: float = 1.0
    reasoning: str | None = None


class ConfidenceFeatures(BaseModel):
    intent_id: int
    reply_length: float          # normalised word count
    contains_pricing: float      # 0.0 or 1.0
    contains_refund: float       # 0.0 or 1.0
    product_mentioned: float     # 0.0 or 1.0
    thread_depth: float          # normalised
    wiki_hit_count: float        # normalised
    feedback_log_match_score: float  # cosine sim
    draft_length_ratio: float    # draft_tokens / expected
    customer_sentiment: float    # -1, 0, or 1

    def to_list(self) -> list[float]:
        return [
            self.intent_id,
            self.reply_length,
            self.contains_pricing,
            self.contains_refund,
            self.product_mentioned,
            self.thread_depth,
            self.wiki_hit_count,
            self.feedback_log_match_score,
            self.draft_length_ratio,
            self.customer_sentiment,
        ]


class Draft(BaseModel):
    id: int | None = None
    thread_id: str
    draft_text: str
    confidence_score: float
    intent: IntentLabel
    features: ConfidenceFeatures
    status: DraftStatus = DraftStatus.PENDING
    final_text: str | None = None
    edit_distance: float | None = None
    label: float | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    resolved_at: datetime | None = None


class TokenLog(BaseModel):
    id: int | None = None
    thread_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    step: str  # 'intent_router' | 'llm_draft' | 'wiki_ingest'
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    context_snapshot: str | None = None  # JSON blob


class FeedbackRecord(BaseModel):
    id: int | None = None
    draft_id: int
    action: FeedbackAction
    original_text: str
    final_text: str
    label: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
