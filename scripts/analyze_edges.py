"""
scripts/analyze_edges.py
Diagnose explicit vs implicit cross-references in the wiki.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

WIKI = Path(os.getenv("WIKI_PATH", "wiki/pages"))
LINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")

pages = []
for md in sorted(WIKI.rglob("*.md")):
    if md.name in ("schema.md", "index.md"):
        continue
    pages.append({
        "id": md.stem,
        "category": md.parent.name,
        "content": md.read_text(encoding="utf-8"),
    })

page_ids = {p["id"] for p in pages}

# Build a map of friendly name → page id, also include id itself
# Friendly names come from H1 in markdown
name_map = {}
for p in pages:
    name_map[p["id"]] = p["id"]
    # Get H1 title
    m = re.search(r"^#\s+(.+)$", p["content"], re.MULTILINE)
    if m:
        title = m.group(1).strip()
        # Title might be "EmailPilot" or "Lumenx: Small Tools, Big Leverage"
        # Use the first word/phrase
        name_map[title.lower()] = p["id"]
        # Also without spaces, lowercase
        clean = re.sub(r"[^a-zA-Z0-9]", "", title).lower()
        name_map[clean] = p["id"]

print("=" * 70)
print(f"Loaded {len(pages)} pages")
print(f"Name aliases (page_id <- name): {len(name_map)}")
print("=" * 70)

# 1. EXPLICIT EDGES: [[wiki-link]]
explicit_edges = []
ghost_edges    = []
for p in pages:
    for link in LINK_RE.findall(p["content"]):
        target = link.strip()
        if target in page_ids:
            explicit_edges.append((p["id"], target, "explicit"))
        else:
            ghost_edges.append((p["id"], target, "ghost"))

print(f"\nExplicit [[wiki-link]] edges: {len(explicit_edges)}")
for src, tgt, _ in explicit_edges[:10]:
    print(f"  {src} -> {tgt}")
if len(explicit_edges) > 10:
    print(f"  ... ({len(explicit_edges) - 10} more)")

print(f"\nGhost [[wiki-link]] edges (target page doesn't exist): {len(ghost_edges)}")
ghost_targets = {tgt for _, tgt, _ in ghost_edges}
print(f"  Unique ghost targets: {len(ghost_targets)}")
for t in sorted(ghost_targets)[:10]:
    print(f"  - {t}")

# 2. IMPLICIT EDGES: plain-text mention of another product id (case-insensitive, word boundary)
# Build a regex per product
product_ids = [p["id"] for p in pages if p["category"] == "products"]
print(f"\n\nProduct pages found: {len(product_ids)}")
print(f"  {', '.join(product_ids)}")

# For each product, also get its H1 name (e.g. "EmailPilot")
product_names = {}
for p in pages:
    if p["category"] != "products":
        continue
    m = re.search(r"^#\s+(.+?)$", p["content"], re.MULTILINE)
    title = m.group(1).strip() if m else p["id"]
    # The friendly name is the title without descriptive parts
    title = title.split(":")[0].strip()  # in case of "EmailPilot: tagline"
    product_names[p["id"]] = title

print(f"\nProduct friendly names:")
for pid, name in product_names.items():
    print(f"  {pid:18s} -> {name}")

# Build mention regex
implicit_edges = []
mention_counts = {}
for p in pages:
    txt = p["content"]
    for other_id, other_name in product_names.items():
        if other_id == p["id"]:
            continue
        # Case-insensitive word-boundary match for product name
        pattern = re.compile(r"\b" + re.escape(other_name) + r"\b", re.IGNORECASE)
        matches = pattern.findall(txt)
        if matches:
            implicit_edges.append((p["id"], other_id, "mention"))
            key = (p["id"], other_id)
            mention_counts[key] = len(matches)

print(f"\n\nImplicit plain-text mention edges: {len(implicit_edges)}")
for src, tgt, _ in implicit_edges[:15]:
    cnt = mention_counts.get((src, tgt), 0)
    print(f"  {src:18s} -> {tgt:18s}  ({cnt} mentions)")
if len(implicit_edges) > 15:
    print(f"  ... ({len(implicit_edges) - 15} more)")

# 3. Total summary
print("\n" + "=" * 70)
print("EDGE SUMMARY")
print("=" * 70)
print(f"  Explicit [[link]]  (target exists):   {len(explicit_edges):4d}")
print(f"  Ghost [[link]]     (target missing):  {len(ghost_edges):4d}")
print(f"  Implicit mentions  (cross-product):   {len(implicit_edges):4d}")
print(f"  TOTAL real edges (explicit+implicit): {len(explicit_edges) + len(implicit_edges):4d}")
print()

# 4. Show pages with the most outgoing connections
outgoing = {}
for p in pages:
    outgoing[p["id"]] = 0
for src, _, _ in explicit_edges + implicit_edges:
    outgoing[src] = outgoing.get(src, 0) + 1

print("Top 10 most-connected pages (outgoing):")
for pid, cnt in sorted(outgoing.items(), key=lambda x: -x[1])[:10]:
    print(f"  {pid:18s}  {cnt} outgoing")
