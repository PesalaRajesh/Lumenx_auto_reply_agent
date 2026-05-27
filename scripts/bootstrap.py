"""
scripts/bootstrap.py
One-time setup: initialise DB, fetch all LumenX data, build wiki.
Run before starting the polling daemon.

Usage:
    python scripts/bootstrap.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import track

load_dotenv()
console = Console()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")

from data.database import Database
from data.lumenx_client import LumenXClient
from wiki.wiki_builder import WikiBuilder


async def main() -> None:
    base_url    = os.environ["LUMENX_BASE_URL"]
    admin_token = os.environ["LUMENX_ADMIN_TOKEN"]
    db_path     = os.getenv("DB_PATH", "data/agent.db")
    wiki_path   = os.getenv("WIKI_PATH", "wiki/pages")
    api_key     = os.environ["ANTHROPIC_API_KEY"]

    console.rule("[bold green]LumenX Agent Bootstrap")

    # 1. Init DB
    console.print("[cyan]1/4[/] Initialising database...")
    db = Database(db_path)
    await db.init()
    console.print(f"    ✓ Database ready at {db_path}")

    async with LumenXClient(base_url, admin_token) as lumenx:
        # 2. Export all data & store products cache
        console.print("[cyan]2/4[/] Exporting LumenX data...")
        export_data = await lumenx.export_all()
        products = await lumenx.get_products()

        # Cache products to DB
        import aiosqlite
        async with aiosqlite.connect(db_path) as db_conn:
            for product in products:
                await db_conn.execute(
                    "INSERT OR REPLACE INTO products_cache (id, name, raw_json) VALUES (?, ?, ?)",
                    (product.get("id"), product.get("name"), json.dumps(product)),
                )
            await db_conn.commit()
        console.print(f"    ✓ Cached {len(products)} products")

        # Save full export for training bootstrap
        export_path = Path("data/export_bootstrap.json")
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(json.dumps(export_data, indent=2))
        console.print(f"    ✓ Export saved to {export_path}")

        # 3. Build wiki
        console.print("[cyan]3/4[/] Building LLM wiki...")
        Path(wiki_path).mkdir(parents=True, exist_ok=True)

        # Write wiki schema
        schema_path = Path(wiki_path) / "schema.md"
        if not schema_path.exists():
            schema_path.write_text(WIKI_SCHEMA, encoding="utf-8")

        anthropic_client = anthropic.AsyncAnthropic(api_key=api_key)
        wiki_builder = WikiBuilder(anthropic_client, lumenx, db, wiki_path)
        await wiki_builder.bootstrap()
        console.print("    ✓ Wiki pages generated")

        # 4. Stats check
        console.print("[cyan]4/4[/] Verifying connection...")
        stats = await lumenx.get_stats()
        console.print(f"    ✓ LumenX stats: {stats}")

    console.rule("[bold green]Bootstrap Complete!")
    console.print("\n[bold]Next steps:[/]")
    console.print("  1. Start the dashboard:  [cyan]uvicorn dashboard.main:app --reload --port 8080[/]")
    console.print("  2. Start inbox polling:  [cyan]python scripts/poll_inbox.py[/]")


WIKI_SCHEMA = """# LumenX Wiki Schema

## Conventions

- Each page covers ONE entity (product, policy, or FAQ topic).
- Never hard-code specific prices — write "(see current pricing in product JSON)".
- Never hard-code refund windows — write "(see current policy in product JSON)".
- Pages MUST include a "## Common Customer Questions" section.
- Cross-reference related pages with [[page_name]] links.

## Page Format

```markdown
# [Page Title]

## Overview
Brief description (2-3 sentences).

## Key Features / Details
Bullet points.

## Important Notes
Any caveats, especially around pricing/refund (no hard-coded values).

## Common Customer Questions

**Q: ...**
A: ...

**Q: ...**
A: ...
```
"""

if __name__ == "__main__":
    asyncio.run(main())
