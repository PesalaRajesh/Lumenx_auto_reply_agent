"""
dashboard/routers/wiki_explorer.py
Wiki knowledge-graph visualization + Q&A endpoints.

Routes:
  GET  /api/wiki/graph         -> nodes + edges for the knowledge graph
  GET  /api/wiki/page/{id}     -> full markdown for a single page
  POST /api/wiki/query         -> RAG-style Q&A with cited sources
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import anthropic
import nltk
from fastapi import APIRouter, HTTPException, Request
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# Ensure NLTK tokenizer is available
for resource in ("tokenizers/punkt", "tokenizers/punkt_tab"):
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource.split("/", 1)[1], quiet=True)

router = APIRouter(prefix="/api/wiki", tags=["wiki"])

WIKI_PATH = Path(os.getenv("WIKI_PATH", "wiki/pages"))
LINK_PATTERN = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
SKIP_FILES = {"schema.md", "index.md"}


def _normalize_id(s: str) -> str:
    """Normalize a link target / page id for case-insensitive matching.

    `[[InvoiceFlow]]` → 'invoiceflow' which matches page id `invoiceflow`.
    Strips whitespace, underscores, and dashes; lowercases.
    """
    return re.sub(r"[\s_\-]+", "", s).lower()

_HAIKU_MODEL = "claude-haiku-4-5"
_HAIKU_INPUT_COST = 0.80
_HAIKU_OUTPUT_COST = 4.00

# Cache of parsed pages, rebuilt on every request (cheap; 21 small files)
def _scan_wiki() -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    if not WIKI_PATH.exists():
        return pages
    for md_file in sorted(WIKI_PATH.rglob("*.md")):
        if md_file.name in SKIP_FILES:
            continue
        try:
            content = md_file.read_text(encoding="utf-8")
        except OSError:
            continue
        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else md_file.stem
        category = md_file.parent.name
        links = [link.strip() for link in LINK_PATTERN.findall(content)]
        pages.append({
            "id":       md_file.stem,
            "title":    title,
            "category": category,
            "path":     str(md_file.relative_to(WIKI_PATH)).replace("\\", "/"),
            "content":  content,
            "links":    links,
        })
    return pages


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/graph")
async def get_graph() -> dict[str, Any]:
    """Return the wiki as a node-edge graph for vis-network.

    Edge resolution is case- and punctuation-insensitive:
    `[[InvoiceFlow]]` resolves to the page id `invoiceflow`. Duplicate edges
    from the same source to the same target are merged into one weighted edge.
    """
    pages = _scan_wiki()

    # Build a normalized lookup so [[InvoiceFlow]] resolves to the 'invoiceflow' page
    id_lookup: dict[str, str] = {_normalize_id(p["id"]): p["id"] for p in pages}
    # Also let the page title normalize to the page id (e.g. [[Lumenx Support SLA]])
    for p in pages:
        id_lookup.setdefault(_normalize_id(p["title"]), p["id"])

    nodes: list[dict[str, Any]] = []
    # (from, to) -> weight (number of references)
    edge_weights: dict[tuple[str, str], int] = {}
    # Map ghost-target normalized-key -> display label (preserve first-seen casing)
    ghost_label: dict[str, str] = {}

    for page in pages:
        word_count = len(page["content"].split())
        nodes.append({
            "id":         page["id"],
            "label":      page["title"],
            "group":      page["category"],
            "ghost":      False,
            "value":      max(word_count // 30, 6),
            "path":       page["path"],
            "wordCount":  word_count,
            "linkCount":  len(page["links"]),
        })

        for raw_link in page["links"]:
            norm = _normalize_id(raw_link)
            if not norm:
                continue
            target = id_lookup.get(norm)
            if target is None:
                # Unresolved -> ghost node, keyed by normalized form
                target = f"ghost::{norm}"
                ghost_label.setdefault(norm, raw_link)
            # Don't create self-loops
            if target == page["id"]:
                continue
            key = (page["id"], target)
            edge_weights[key] = edge_weights.get(key, 0) + 1

    # Emit ghost nodes
    for norm, label in ghost_label.items():
        nodes.append({
            "id":    f"ghost::{norm}",
            "label": label.strip(),
            "group": "ghost",
            "ghost": True,
            "value": 4,
        })

    # Emit edges (deduped + weighted)
    edges: list[dict[str, Any]] = []
    for (src, dst), weight in edge_weights.items():
        edges.append({
            "from":   src,
            "to":     dst,
            "weight": weight,
        })

    real_edges  = sum(1 for e in edges if not e["to"].startswith("ghost::"))
    ghost_edges = len(edges) - real_edges

    return {
        "nodes":      nodes,
        "edges":      edges,
        "pageCount":  len(pages),
        "ghostCount": len(ghost_label),
        "realEdgeCount":  real_edges,
        "ghostEdgeCount": ghost_edges,
    }


@router.get("/page/{page_id}")
async def get_page(page_id: str) -> dict[str, Any]:
    """Return full markdown content for one page."""
    if not WIKI_PATH.exists():
        raise HTTPException(404, "Wiki directory missing")
    safe_id = re.sub(r"[^a-zA-Z0-9_\-]", "", page_id)
    matches = list(WIKI_PATH.rglob(f"{safe_id}.md"))
    if not matches:
        raise HTTPException(404, f"Page '{page_id}' not found")
    md_file = matches[0]
    content = md_file.read_text(encoding="utf-8")
    title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    return {
        "id":      page_id,
        "title":   title_match.group(1).strip() if title_match else page_id,
        "path":    str(md_file.relative_to(WIKI_PATH)).replace("\\", "/"),
        "content": content,
        "category": md_file.parent.name,
    }


@router.post("/query")
async def query_wiki(request: Request) -> dict[str, Any]:
    """RAG-style Q&A: BM25 retrieval -> Claude Haiku -> answer + sources."""
    body = await request.json()
    question = (body.get("question") or "").strip()
    if not question:
        raise HTTPException(400, "Field 'question' is required")
    if len(question) > 1000:
        raise HTTPException(400, "Question too long")

    pages = _scan_wiki()
    if not pages:
        raise HTTPException(500, "Wiki has no pages — run scripts/bootstrap.py first")

    # ── BM25 retrieval ─────────────────────────────────────────────
    tokenised = [nltk.word_tokenize(p["content"].lower()) for p in pages]
    bm25 = BM25Okapi(tokenised)
    scores = bm25.get_scores(nltk.word_tokenize(question.lower()))
    ranked = sorted(zip(scores, pages), key=lambda x: x[0], reverse=True)
    top = [(score, page) for score, page in ranked[:3] if score > 0]

    if not top:
        return {
            "answer": "I couldn't find anything in the wiki relevant to that question. Try rephrasing or asking about a specific product.",
            "sources": [],
            "tokens": {"input": 0, "output": 0, "cost_usd": 0.0},
        }

    # ── Build context for Claude ──────────────────────────────────
    context_parts: list[str] = []
    for i, (score, page) in enumerate(top, start=1):
        context_parts.append(
            f"=== Source [{i}] — {page['title']} (file: {page['path']}, BM25={score:.2f}) ===\n"
            f"{page['content']}"
        )
    context = "\n\n".join(context_parts)

    system_prompt = (
        "You are a precise Q&A assistant for the LumenX product wiki. "
        "Answer ONLY using the provided sources. "
        "After your answer, on a new line, write 'Sources: [1]' or 'Sources: [1, 2]' citing which source(s) you used. "
        "If the sources do not contain the answer, say so explicitly — DO NOT invent facts. "
        "NEVER fabricate specific prices, refund windows, or trial periods — if the source says 'see current pricing in product JSON', "
        "tell the user you don't have the exact figure but explain what data IS available. "
        "Be concise (under 200 words). Use plain prose, no markdown headers."
    )

    user_message = f"SOURCES FROM WIKI:\n\n{context}\n\n---\n\nQUESTION: {question}"

    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = await client.messages.create(
        model=_HAIKU_MODEL,
        max_tokens=600,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    answer = resp.content[0].text.strip()
    input_tokens = resp.usage.input_tokens
    output_tokens = resp.usage.output_tokens
    cost = (input_tokens * _HAIKU_INPUT_COST + output_tokens * _HAIKU_OUTPUT_COST) / 1_000_000

    # ── Parse cited sources from answer (e.g. "Sources: [1, 2]") ─────
    cited_indices: set[int] = set()
    cite_match = re.search(r"Sources?:\s*\[([0-9,\s]+)\]", answer, re.IGNORECASE)
    if cite_match:
        for tok in cite_match.group(1).split(","):
            tok = tok.strip()
            if tok.isdigit():
                cited_indices.add(int(tok))

    sources_payload: list[dict[str, Any]] = []
    for i, (score, page) in enumerate(top, start=1):
        sources_payload.append({
            "index":     i,
            "id":        page["id"],
            "title":     page["title"],
            "path":      page["path"],
            "category":  page["category"],
            "bm25":      round(score, 2),
            "excerpt":   page["content"][:300] + ("..." if len(page["content"]) > 300 else ""),
            "cited":     i in cited_indices or not cited_indices,  # if model didn't cite, mark all
        })

    logger.info(
        "Wiki query (%d chars) -> %d sources, %d cited, $%.5f",
        len(question), len(sources_payload),
        sum(1 for s in sources_payload if s["cited"]), cost,
    )

    return {
        "answer":   answer,
        "sources":  sources_payload,
        "tokens":   {
            "input":    input_tokens,
            "output":   output_tokens,
            "cost_usd": round(cost, 6),
        },
    }
