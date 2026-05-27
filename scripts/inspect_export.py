"""Inspect the seeded export to understand bootstrap data structure."""
import json
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

p = Path("data/export_bootstrap.json")
with p.open(encoding="utf-8") as f:
    data = json.load(f)

print("Top-level keys:", list(data.keys()))
threads = data.get("threads") or data.get("conversations") or []
print(f"Threads: {len(threads)}")

if threads:
    t0 = threads[0]
    print(f"First thread keys: {list(t0.keys())}")
    msgs = t0.get("messages", [])
    print(f"First thread message count: {len(msgs)}")
    if msgs:
        print(f"First message keys: {list(msgs[0].keys())}")

    print("\n--- Sample exchanges (first 3 threads with admin replies) ---")
    seeded = [
        t for t in threads
        if any(m.get("role") == "admin" for m in t.get("messages", []))
        and any(m.get("role") == "customer" for m in t.get("messages", []))
    ]
    print(f"Threads with both customer+admin messages: {len(seeded)}")
    for t in seeded[:3]:
        print(f"\nThread {t.get('id')} (intent={t.get('intent')}, product={t.get('product_id')})")
        for m in t.get("messages", []):
            role = m.get("role", "?")
            text = (m.get("text") or "")[:200].replace("\n", " ")
            print(f"  [{role:8s}] {text}")

    # Stats on seeded threads
    print("\n--- Stats for ALL seeded threads ---")
    intent_counts = {}
    product_counts = {}
    customer_q_only = 0
    has_admin_reply = 0
    for t in seeded:
        intent_counts[t.get("intent", "?")] = intent_counts.get(t.get("intent", "?"), 0) + 1
        product_counts[t.get("product_id") or "(none)"] = product_counts.get(t.get("product_id") or "(none)", 0) + 1
        msgs = t.get("messages", [])
        has_admin = any(m.get("role") == "admin" for m in msgs)
        if has_admin:
            has_admin_reply += 1
        else:
            customer_q_only += 1

    print(f"  Seeded threads:       {len(seeded)}")
    print(f"  ...with admin reply:  {has_admin_reply}  <-- usable for training")
    print(f"  ...customer-only:     {customer_q_only}")
    print(f"  Intent distribution:  {sorted(intent_counts.items(), key=lambda x: -x[1])}")
