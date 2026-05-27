"""
scripts/e2e_chat_test.py
Headless-browser E2E test of the /chat page using Playwright.

Verifies:
  1. /chat loads with no JS errors
  2. The chat proxy is reachable (no 'Failed to fetch')
  3. Starting a new thread succeeds
  4. Posting a customer message succeeds
  5. The message appears in the chat feed

Usage:
    python scripts/e2e_chat_test.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from playwright.async_api import async_playwright, ConsoleMessage  # type: ignore

OUT_DIR = Path(__file__).parent.parent / "data" / "e2e_screenshots"
OUT_DIR.mkdir(parents=True, exist_ok=True)


async def main() -> int:
    failures: list[str] = []
    console_errors: list[str] = []
    page_errors: list[str] = []

    async with async_playwright() as pw:
        # Use system-installed Microsoft Edge (avoids the playwright-managed
        # Chromium download which fails on this machine due to a corporate
        # cert chain that Node.js doesn't trust).
        browser = await pw.chromium.launch(headless=True, channel="msedge")
        context = await browser.new_context(viewport={"width": 1280, "height": 800})
        page = await context.new_page()

        page.on("console", lambda msg: console_errors.append(f"[{msg.type}] {msg.text}")
                if msg.type in ("error", "warning") else None)
        page.on("pageerror", lambda exc: page_errors.append(str(exc)))

        # ── 1. Load /chat ─────────────────────────────────────
        print("1. Loading http://127.0.0.1:8080/chat ...")
        resp = await page.goto("http://127.0.0.1:8080/chat", wait_until="networkidle", timeout=15000)
        if not resp or not resp.ok:
            failures.append(f"/chat returned {resp.status if resp else 'no response'}")
        await page.screenshot(path=str(OUT_DIR / "01_loaded.png"))
        print(f"   ✓ /chat HTTP {resp.status if resp else 'n/a'}, screenshot saved")

        # ── 2. Pre-fill username ──────────────────────────────
        print("2. Setting username...")
        await page.fill("#username-input", "playwright_e2e")
        await page.locator("#username-input").press("Tab")
        print("   ✓ username set")

        # ── 3. Click 'Start new chat' ─────────────────────────
        print("3. Clicking 'Start new chat'...")
        await page.click("#new-thread-btn")

        # Wait until thread-id stops being 'no thread selected'
        try:
            await page.wait_for_function(
                """() => {
                    const el = document.getElementById('current-thread-id');
                    return el && el.textContent && el.textContent.startsWith('live-');
                }""",
                timeout=10000,
            )
            tid_text = await page.text_content("#current-thread-id")
            print(f"   ✓ Thread created: {tid_text}")
        except Exception as exc:
            failures.append(f"Thread creation timeout: {exc}")
            await page.screenshot(path=str(OUT_DIR / "fail_no_thread.png"))
            print(f"   ✗ Thread not created — see fail_no_thread.png")

        await page.screenshot(path=str(OUT_DIR / "02_thread_created.png"))

        # ── 4. Type and send a customer message ──────────────
        print("4. Sending a customer message...")
        test_msg = "Playwright probe: what is the EmailPilot Pro price?"
        await page.fill("#msg-input", test_msg)
        await page.click("#send-btn")

        # Wait for the customer bubble to appear in the feed
        try:
            await page.wait_for_function(
                f"""() => {{
                    const bubbles = document.querySelectorAll('.msg.customer .bubble');
                    for (const b of bubbles) {{
                        if (b.textContent.includes({test_msg!r})) return true;
                    }}
                    return false;
                }}""",
                timeout=10000,
            )
            print("   ✓ Customer message rendered")
        except Exception as exc:
            failures.append(f"Message did not appear in feed: {exc}")
            print(f"   ✗ Message not visible in feed")

        await page.screenshot(path=str(OUT_DIR / "03_message_sent.png"))

        # ── 5. Verify 'Lumi is drafting…' indicator shows up ─
        print("5. Checking 'Lumi is drafting…' indicator...")
        try:
            await page.wait_for_selector(".awaiting", timeout=5000)
            print("   ✓ Drafting indicator visible (agent should respond when poller runs)")
        except Exception:
            failures.append("'Lumi is drafting…' indicator never appeared")
            print("   ✗ No drafting indicator")

        await page.screenshot(path=str(OUT_DIR / "04_awaiting_reply.png"))

        # ── 6. Capture connection status ─────────────────────
        conn = await page.text_content("#conn-status")
        print(f"6. Connection status badge: {conn.strip() if conn else '(empty)'}")
        if conn and "Reconnecting" in conn:
            failures.append("Connection status is 'Reconnecting' — proxy may be down")

        await browser.close()

    # ── Report ───────────────────────────────────────────────
    print()
    print("=" * 60)
    if console_errors:
        print(f"Console messages ({len(console_errors)}):")
        for e in console_errors[:10]:
            print(f"  {e}")
    if page_errors:
        print(f"Page JS errors ({len(page_errors)}):")
        for e in page_errors[:5]:
            print(f"  {e}")
        failures.append(f"{len(page_errors)} JS exceptions")
    if failures:
        print(f"❌ FAIL ({len(failures)} issue{'s' if len(failures) != 1 else ''})")
        for f in failures:
            print(f"   - {f}")
        return 1
    print("✅ PASS — all 6 checks passed")
    print(f"Screenshots in: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
