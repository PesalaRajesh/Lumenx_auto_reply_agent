"""
wiki/wiki_query.py
BM25-based retrieval over local wiki markdown pages.
"""
from __future__ import annotations

import logging
from pathlib import Path

import nltk
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# Ensure NLTK tokenizer data is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)


class WikiQuery:
    """Indexes wiki markdown files and retrieves top-K pages by BM25 score."""

    def __init__(self, wiki_pages_dir: str | Path) -> None:
        self._pages_dir = Path(wiki_pages_dir)
        self._index: BM25Okapi | None = None
        self._documents: list[dict[str, str]] = []
        self._build_index()

    def _build_index(self) -> None:
        """Scan wiki pages directory and build BM25 index."""
        self._documents = []
        tokenized_corpus: list[list[str]] = []

        for md_file in sorted(self._pages_dir.rglob("*.md")):
            if md_file.name in ("schema.md", "index.md"):
                continue
            content = md_file.read_text(encoding="utf-8")
            title = md_file.stem.replace("_", " ").title()
            self._documents.append({"title": title, "path": str(md_file), "content": content})
            tokens = nltk.word_tokenize(content.lower())
            tokenized_corpus.append(tokens)

        if tokenized_corpus:
            self._index = BM25Okapi(tokenized_corpus)
            logger.info("Wiki index built: %d pages indexed", len(self._documents))
        else:
            logger.warning("No wiki pages found in %s", self._pages_dir)

    async def query(self, query_text: str, top_k: int = 3) -> list[dict[str, str]]:
        """Return the top-K wiki pages most relevant to query_text."""
        if self._index is None or not self._documents:
            return []

        tokens = nltk.word_tokenize(query_text.lower())
        scores = self._index.get_scores(tokens)

        ranked = sorted(
            zip(scores, self._documents), key=lambda x: x[0], reverse=True
        )
        results = []
        for score, doc in ranked[:top_k]:
            if score > 0:
                results.append({
                    "title": doc["title"],
                    "content": doc["content"][:2000],  # Truncate individual pages
                    "score": score,
                })

        return results

    def refresh(self) -> None:
        """Rebuild the index (call after wiki pages are updated)."""
        self._build_index()
