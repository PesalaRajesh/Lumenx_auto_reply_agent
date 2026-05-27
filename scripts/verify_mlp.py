"""Verify the trained Confidence Net distinguishes good/bad drafts."""
from __future__ import annotations

import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.confidence_net import ConfidenceNet
from data.models import ConfidenceFeatures, IntentLabel
from training.feature_extractor import extract_features

cn = ConfidenceNet()
print(f"Confidence Net loaded — trained = {cn.is_trained}")
print()

# Sample inputs spanning realistic ranges
samples = [
    ("Terse, factual pricing reply (short, no pricing keyword)", extract_features(
        draft_text="Hi Lucas! For PixelDeck, starter is free, pro is usd 9 per month. Let me know if you'd like a hand.",
        intent=IntentLabel.PRICING, thread_depth=1, wiki_hit_count=2,
        feedback_log_match_score=0.7, customer_message="how much is pixeldeck?",
        draft_tokens=25,
    )),
    ("Verbose, multi-section product info reply", extract_features(
        draft_text=(
            "Hi there! Great question about our products. EmailPilot is fantastic for managing email volume. "
            "It includes tone-matched drafts, attachment context, per-thread summaries, snooze and follow-up reminders. "
            "Pricing starts at usd 9 per month for Starter, with Pro and Team plans available. There's a 14-day trial. "
            "I'd recommend Pro for most professionals. Other products like TaskGrid and NoteHub also exist. "
            "Let me know what you're looking for and I can give a more specific recommendation."
        ),
        intent=IntentLabel.PRODUCT_INFO, thread_depth=2, wiki_hit_count=3,
        feedback_log_match_score=0.3, customer_message="what products do you have?",
        draft_tokens=110,
    )),
    ("Truncated / abrupt reply (bad)", extract_features(
        draft_text="Hi! For PixelDeck...",
        intent=IntentLabel.PRICING, thread_depth=1, wiki_hit_count=1,
        feedback_log_match_score=0.1, customer_message="how much?",
        draft_tokens=5,
    )),
    ("Reply containing pricing + refund keywords (more sensitive)", extract_features(
        draft_text="Pro is usd 19 per month. Full refund within 14 days of first purchase.",
        intent=IntentLabel.PRICING, thread_depth=1, wiki_hit_count=2,
        feedback_log_match_score=0.5, customer_message="price and refund?",
        draft_tokens=18,
    )),
    ("Negative-sentiment complaint reply", extract_features(
        draft_text="Hi Ravi, sorry for the trouble! Let's sort this out together.",
        intent=IntentLabel.COMPLAINT, thread_depth=4, wiki_hit_count=2,
        feedback_log_match_score=0.4, customer_message="I'm so frustrated, nothing works",
        draft_tokens=15,
    )),
]

print(f"{'Sample':<60s} {'Score':>7s}  {'Decision':>14s}")
print("-" * 86)
THRESHOLD = 0.75
for label, feats in samples:
    score = cn.predict(feats)
    decision = "AUTO-SEND" if score >= THRESHOLD else "human review"
    print(f"{label[:60]:<60s}   {score:.3f}   {decision:>14s}")

print()
print(f"Threshold: {THRESHOLD}")
print(f"Untrained baseline would be 0.500 for all — trained MLP should now vary.")
