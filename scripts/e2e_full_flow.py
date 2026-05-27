"""
scripts/e2e_full_flow.py
Full end-to-end verification:
  1. Start the agent poller daemon as a subprocess
  2. Open /chat in a headless Edge browser
  3. Create a thread + send a customer message
  4. Wait for the agent to draft + (auto-send or reach pending) the reply
  5. Confirm the reply appears in the chat feed (if auto-sent) OR confirm a
     pending draft was created (if the MLP held it for review)
  6. Screenshot the final state
  7. Kill the poller cleanly
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from playwright.async_api import async_playwright  # type: ignore

OUT_DIR = ROOT / "data" / "e2e_screenshots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

POLLER_TIMEOUT = 90  # seconds to wait for agent reply
MESSAGE = "What is the price of EmailPilot Pro per month, and is there a free trial?"


async def open_chat_and_send(page) -> tuple[str, int]:
    """Open /chat, create a thread, send a message. Returns (thread_id, msg_count_before_reply)."""
    print("→ Loading /chat...")
    await page.goto("http://127.0.0.1:8080/chat", wait_until="networkidle", timeout=15000)
    await page.fill("#username-input", "playwright_full")
    await page.locator("#username-input").press("Tab")
    await page.click("#new-thread-btn")
    await page.wait_for_function(
        """() => {
            const el = document.getElementById('current-thread-id');
            return el && el.textContent.startsWith('live-');
        }""",
        timeout=10000,
    )
    tid = (await page.text_content("#current-thread-id")).strip()
    print(f"→ Thread created: {tid}")

    await page.fill("#msg-input", MESSAGE)
    await page.click("#send-btn")
    # Wait for the customer bubble
    await page.wait_for_function(
        f"() => document.querySelectorAll('.msg.customer .bubble').length >= 1",
        timeout=5000,
    )
    print(f"→ Customer message sent")
    return tid, 1


async def wait_for_agent_response(page, thread_id: str, timeout_s: int) -> dict:
    """Watch the page until an admin bubble appears OR pending-drafts has this thread."""
    start = time.time()
    while time.time() - start < timeout_s:
        # Check for admin bubble in the open chat feed
        admin_count = await page.evaluate(
            "() => document.querySelectorAll('.msg.admin .bubble').length"
        )
        if admin_count >= 1:
            return {"outcome": "auto_sent", "elapsed_s": time.time() - start}

        # Otherwise check our own dashboard's pending drafts
        pending = await page.evaluate(
            f"""async () => {{
                try {{
                    const r = await fetch('/api/pending-drafts');
                    const drafts = await r.json();
                    return drafts.filter(d => d.thread_id === {thread_id!r});
                }} catch (e) {{ return []; }}
            }}"""
        )
        if pending:
            return {
                "outcome": "human_review",
                "elapsed_s": time.time() - start,
                "draft": pending[0],
            }

        await asyncio.sleep(2)

    return {"outcome": "timeout", "elapsed_s": time.time() - start}


async def main() -> int:
    # ── 1. Start the poller subprocess ─────────────────────────
    print("=" * 64)
    print(" E2E FULL FLOW: customer chat -> agent draft -> reply")
    print("=" * 64)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONIOENCODING"] = "utf-8"
    log_path = ROOT / "data" / "e2e_poller.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"→ Starting poller, log -> {log_path}")
    poller = subprocess.Popen(
        [str(ROOT / "venv" / "Scripts" / "python.exe"), str(ROOT / "scripts" / "poll_inbox.py")],
        cwd=ROOT, env=env,
        stdout=log_path.open("w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
    )

    try:
        # Give the poller a moment to spin up
        await asyncio.sleep(2)

        # ── 2. Drive the browser ─────────────────────────────
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True, channel="msedge")
            context = await browser.new_context(viewport={"width": 1280, "height": 800})
            page = await context.new_page()

            tid, _ = await open_chat_and_send(page)
            await page.screenshot(path=str(OUT_DIR / "full_01_message_sent.png"))

            print(f"→ Waiting up to {POLLER_TIMEOUT}s for the agent...")
            result = await wait_for_agent_response(page, tid, POLLER_TIMEOUT)

            await page.screenshot(path=str(OUT_DIR / "full_02_final.png"))

            await browser.close()

    finally:
        # ── 3. Stop the poller ────────────────────────────────
        print("→ Stopping poller...")
        poller.terminate()
        try:
            poller.wait(timeout=5)
        except subprocess.TimeoutExpired:
            poller.kill()

    # ── 4. Report ───────────────────────────────────────────
    print()
    print("=" * 64)
    if result["outcome"] == "auto_sent":
        print(f"✅ AUTO-SEND succeeded in {result['elapsed_s']:.1f}s — agent reply visible in chat")
        return 0
    elif result["outcome"] == "human_review":
        d = result["draft"]
        print(f"✅ HUMAN REVIEW: draft #{d['id']} created in {result['elapsed_s']:.1f}s")
        print(f"   intent={d['intent']}  confidence={d['confidence_score']:.3f}")
        print(f"   Visit http://127.0.0.1:8080/inbox to approve/edit/reject.")
        return 0
    else:
        print(f"❌ TIMEOUT after {result['elapsed_s']:.1f}s — no admin reply and no pending draft")
        print(f"   Poller log: {log_path}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
