"""
wiki/wiki_builder.py
Bootstraps and updates the LLM wiki from the LumenX API.
Follows Karpathy's LLM-maintained wiki approach.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import anthropic

from data.lumenx_client import LumenXClient
from data.database import Database

logger = logging.getLogger(__name__)

_HAIKU_MODEL = "claude-haiku-4-5"
_HAIKU_INPUT_COST  = 0.80
_HAIKU_OUTPUT_COST = 4.00

_INGEST_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "wiki_ingest_prompt.md"


class WikiBuilder:
    """Builds and maintains the LLM wiki from LumenX product/policy data."""

    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        lumenx: LumenXClient,
        db: Database,
        wiki_pages_dir: str | Path,
    ) -> None:
        self._client = client
        self._lumenx = lumenx
        self._db = db
        self._pages_dir = Path(wiki_pages_dir)
        self._ingest_prompt = _INGEST_PROMPT_PATH.read_text(encoding="utf-8")

    async def bootstrap(self) -> None:
        """
        Full bootstrap: fetch all products + policies, write one wiki page each.
        Safe to re-run — overwrites existing pages.
        """
        logger.info("Starting wiki bootstrap...")

        # Fetch full products payload (products + company-wide policies)
        full       = await self._lumenx.get_products_full()
        products   = full.get("products", [])
        company    = full.get("company", {})
        logger.info("Fetched %d products + company policies", len(products))

        for product in products:
            await self._write_product_page(product)

        # Write a combined company policies page
        await self._write_policies_page(company)

        # Write the wiki index
        self._write_index()
        logger.info("Wiki bootstrap complete.")

    async def _write_product_page(self, product: dict) -> None:
        """Generate and write a wiki page for one product."""
        product_id = product.get("id", "unknown")
        raw_data = json.dumps(product, indent=2)

        content = await self._generate_page(raw_data, context=f"product: {product_id}")

        out_path = self._pages_dir / "products" / f"{product_id}.md"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")
        logger.info("  Wrote wiki page: products/%s.md", product_id)

    async def _write_policies_page(self, company: dict) -> None:
        """Generate a wiki page from the company-wide policies block."""
        raw_data = json.dumps(company, indent=2)
        content = await self._generate_page(raw_data, context="company policies")

        out_path = self._pages_dir / "policies" / "company_policies.md"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")
        logger.info("  Wrote wiki page: policies/company_policies.md")

    async def _generate_page(self, raw_data: str, context: str) -> str:
        """Call Claude Haiku to generate a wiki page from raw data."""
        prompt = self._ingest_prompt.replace("{{raw_data}}", raw_data)

        response = await self._client.messages.create(
            model=_HAIKU_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )

        input_tokens  = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost = (input_tokens * _HAIKU_INPUT_COST + output_tokens * _HAIKU_OUTPUT_COST) / 1_000_000

        await self._db.log_tokens(
            thread_id="wiki_bootstrap",
            step="wiki_ingest",
            model=_HAIKU_MODEL,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )

        return response.content[0].text.strip()

    def _write_index(self) -> None:
        """Write wiki/pages/index.md listing all pages."""
        lines = ["# LumenX Wiki Index\n"]
        for md_file in sorted(self._pages_dir.rglob("*.md")):
            if md_file.name in ("index.md", "schema.md"):
                continue
            rel = md_file.relative_to(self._pages_dir)
            title = md_file.stem.replace("_", " ").title()
            lines.append(f"- [{title}]({rel})")

        index_path = self._pages_dir / "index.md"
        index_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Wiki index updated (%d pages)", len(lines) - 1)
