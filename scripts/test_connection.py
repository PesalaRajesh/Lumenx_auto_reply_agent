"""
scripts/test_connection.py
Sanity check both LumenX + Anthropic API connectivity.
"""
from __future__ import annotations

import asyncio
import os
import sys
import traceback
from pathlib import Path

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic
from dotenv import load_dotenv

load_dotenv()


def info(msg: str) -> None:
    print(msg, flush=True)


async def test_anthropic() -> bool:
    info("\n[Anthropic API]")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or not api_key.startswith("sk-ant-"):
        info("  [FAIL] ANTHROPIC_API_KEY missing or malformed")
        return False
    info(f"  Key prefix: {api_key[:20]}...")

    client = anthropic.AsyncAnthropic(api_key=api_key)
    try:
        info("  Calling claude-haiku-4-5...")
        resp = await client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=50,
            messages=[{"role": "user", "content": "Reply with exactly: PONG"}],
        )
        text = resp.content[0].text.strip()
        info(f"  [OK] Haiku replied: {text}")
        info(f"       Tokens in/out: {resp.usage.input_tokens}/{resp.usage.output_tokens}")

        info("  Calling claude-sonnet-4-6...")
        resp = await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=50,
            messages=[{"role": "user", "content": "Reply with exactly: PONG"}],
        )
        text = resp.content[0].text.strip()
        info(f"  [OK] Sonnet replied: {text}")
        info(f"       Tokens in/out: {resp.usage.input_tokens}/{resp.usage.output_tokens}")
        return True
    except Exception as exc:
        info(f"  [FAIL] Anthropic test failed:")
        info(f"    {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return False


async def test_lumenx() -> bool:
    info("\n[LumenX API]")
    from data.lumenx_client import LumenXClient

    base_url    = os.environ["LUMENX_BASE_URL"]
    admin_token = os.environ["LUMENX_ADMIN_TOKEN"]
    info(f"  Base: {base_url}")

    async with LumenXClient(base_url, admin_token) as lumenx:
        try:
            stats = await lumenx.get_stats()
            info("  [OK] LumenX reachable")
            info(f"    threads:  {stats['threads']}")
            info(f"    messages: {stats['messages']}")

            products = await lumenx.get_products()
            info(f"  [OK] Products fetched: {len(products)}")
            for p in products[:3]:
                info(f"    - {p.get('id')}: {p.get('name')}")
            if len(products) > 3:
                info(f"    ... ({len(products) - 3} more)")
            return True
        except Exception as exc:
            info(f"  [FAIL] LumenX test failed: {exc}")
            traceback.print_exc()
            return False


async def main() -> None:
    info("=" * 60)
    info("Connection Test")
    info("=" * 60)
    a_ok = await test_anthropic()
    l_ok = await test_lumenx()
    info("=" * 60)
    if a_ok and l_ok:
        info("All checks passed - ready to bootstrap.")
    else:
        info("ONE OR MORE CHECKS FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
