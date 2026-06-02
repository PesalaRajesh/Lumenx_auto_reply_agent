"""
dashboard/main.py
FastAPI dashboard for the LumenX Auto-Reply Agent.
Provides: Inbox review, Analytics, Wiki editor, Training log.

Start with:
    uvicorn dashboard.main:app --reload --port 8080
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import aiosqlite
from Levenshtein import ratio as lev_ratio

load_dotenv()

logger = logging.getLogger(__name__)

DB_PATH    = os.getenv("DB_PATH", "data/agent.db")
BASE_URL   = os.getenv("LUMENX_BASE_URL", "https://lumenx-demo.up.railway.app")
ADMIN_TOKEN = os.getenv("LUMENX_ADMIN_TOKEN", "")


async def _run_poller() -> None:
    """Run the inbox poller as a background asyncio task. Restarts on crash."""
    while True:
        try:
            from scripts.poll_inbox import main as poll_main
            await poll_main()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Poller crashed — restarting in 30 s")
            await asyncio.sleep(30)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Init all DB tables (threads, messages, drafts, token_log, feedback, products_cache)
    from data.database import Database
    await Database(DB_PATH).init()

    # Start the inbox poller in the same event loop — shares memory with the web server
    poller = asyncio.create_task(_run_poller())
    logger.info("Inbox poller started as background task")

    yield  # app runs here

    poller.cancel()
    try:
        await poller
    except asyncio.CancelledError:
        pass


app = FastAPI(title="LumenX Agent Dashboard", lifespan=lifespan)
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

# Serve shared CSS / JS from /static (Lumenx design tokens, etc.)
_static_dir = Path(__file__).parent / "static"
_static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=_static_dir), name="static")

# Mount the wiki explorer router
from dashboard.routers import wiki_explorer  # noqa: E402
app.include_router(wiki_explorer.router)

# Proxy LumenX customer endpoints so the local /chat page bypasses CORS
from dashboard.routers import chat_proxy  # noqa: E402
app.include_router(chat_proxy.router)


# ──────────────────────────────────────────────────────────────
# Pages
# ──────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/wiki", response_class=HTMLResponse)
async def wiki_page(request: Request):
    return templates.TemplateResponse(request, "wiki.html")


@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    """Local customer chat UI. Posts to public LumenX API; the poller picks
    up messages and drafts replies the same way it would in production."""
    return templates.TemplateResponse(request, "chat.html")


@app.get("/inbox", response_class=HTMLResponse)
async def inbox_page(request: Request):
    pending = await _get_pending_drafts()
    return templates.TemplateResponse(request, "inbox.html", {"drafts": pending})


@app.get("/analytics", response_class=HTMLResponse)
async def analytics_page(request: Request):
    summary = await _cost_summary()
    return templates.TemplateResponse(request, "analytics.html", {"summary": summary})


# ──────────────────────────────────────────────────────────────
# API endpoints (called by dashboard JS)
# ──────────────────────────────────────────────────────────────

@app.get("/api/pending-drafts")
async def api_pending_drafts():
    return await _get_pending_drafts()


@app.post("/api/drafts/{draft_id}/approve")
async def approve_draft(draft_id: int):
    """Approve a draft as-is and send it."""
    draft = await _get_draft(draft_id)
    if not draft:
        raise HTTPException(404, "Draft not found")

    await _send_reply_to_lumenx(draft["thread_id"], draft["draft_text"], draft["confidence_score"])
    await _resolve_draft(draft_id, "approved", draft["draft_text"], 0.0, 1.0)
    await _save_feedback(draft_id, "approve", draft["draft_text"], draft["draft_text"], 1.0)
    return {"status": "approved"}


@app.post("/api/drafts/{draft_id}/edit")
async def edit_draft(draft_id: int, request: Request):
    """Edit a draft and send the edited version."""
    body = await request.json()
    final_text = body.get("final_text", "").strip()
    if not final_text:
        raise HTTPException(400, "final_text is required")

    draft = await _get_draft(draft_id)
    if not draft:
        raise HTTPException(404, "Draft not found")

    # Compute label from edit distance
    similarity = lev_ratio(draft["draft_text"], final_text)
    label = similarity  # 1.0 = identical, 0.0 = completely different

    await _send_reply_to_lumenx(draft["thread_id"], final_text, draft["confidence_score"])
    await _resolve_draft(draft_id, "edited", final_text, 1.0 - similarity, label)
    await _save_feedback(draft_id, "edit", draft["draft_text"], final_text, label)
    return {"status": "edited", "label": label}


@app.post("/api/drafts/{draft_id}/reject")
async def reject_draft(draft_id: int, request: Request):
    """Reject a draft and optionally send a manual reply."""
    body = await request.json()
    manual_reply = body.get("manual_reply", "").strip()

    draft = await _get_draft(draft_id)
    if not draft:
        raise HTTPException(404, "Draft not found")

    if manual_reply:
        await _send_reply_to_lumenx(draft["thread_id"], manual_reply, None)

    await _resolve_draft(draft_id, "rejected", manual_reply or "", 1.0, 0.0)
    await _save_feedback(draft_id, "reject", draft["draft_text"], manual_reply or "", 0.0)
    return {"status": "rejected"}


@app.get("/api/cost-summary")
async def api_cost_summary():
    return await _cost_summary()


@app.get("/api/token-log")
async def api_token_log(limit: int = 50):
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM token_log ORDER BY timestamp DESC LIMIT ?", (limit,)
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


@app.get("/api/stats")
async def api_stats():
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        pending = (await (await db.execute("SELECT COUNT(*) FROM drafts WHERE status='pending'")).fetchone())[0]
        total   = (await (await db.execute("SELECT COUNT(*) FROM drafts")).fetchone())[0]
        auto    = (await (await db.execute("SELECT COUNT(*) FROM drafts WHERE status='auto_sent'")).fetchone())[0]
        labels  = (await (await db.execute("SELECT COUNT(*) FROM drafts WHERE label IS NOT NULL")).fetchone())[0]
        cost_row = await (await db.execute("SELECT SUM(cost_usd) FROM token_log")).fetchone()
        total_cost = cost_row[0] or 0.0

    return {
        "pending_drafts": pending,
        "total_drafts": total,
        "auto_sent": auto,
        "labelled_examples": labels,
        "total_cost_usd": round(total_cost, 4),
    }


# ──────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────

async def _get_pending_drafts() -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM drafts WHERE status='pending' ORDER BY created_at"
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


async def _get_draft(draft_id: int) -> dict | None:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM drafts WHERE id=?", (draft_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None


async def _resolve_draft(draft_id: int, status: str, final_text: str, edit_distance: float, label: float) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """UPDATE drafts SET status=?, final_text=?, edit_distance=?, label=?,
               resolved_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE id=?""",
            (status, final_text, edit_distance, label, draft_id),
        )
        await db.commit()


async def _save_feedback(draft_id: int, action: str, original: str, final: str, label: float) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO feedback (draft_id, action, original_text, final_text, label) VALUES (?,?,?,?,?)",
            (draft_id, action, original, final, label),
        )
        await db.commit()


async def _send_reply_to_lumenx(thread_id: str, text: str, confidence: float | None) -> None:
    import ssl
    import httpx
    import truststore
    ssl_ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    headers = {"X-Admin-Token": ADMIN_TOKEN, "Content-Type": "application/json"}
    body: dict = {"text": text, "draft_source": "agent"}
    if confidence is not None:
        body["confidence"] = confidence
    async with httpx.AsyncClient(verify=ssl_ctx) as client:
        resp = await client.post(
            f"{BASE_URL}/api/admin/threads/{thread_id}/reply",
            json=body,
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()


async def _cost_summary() -> dict:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """SELECT model,
                      SUM(input_tokens)  AS total_input,
                      SUM(output_tokens) AS total_output,
                      SUM(cost_usd)      AS total_cost,
                      COUNT(*)           AS call_count
               FROM token_log
               GROUP BY model"""
        )
        rows = await cursor.fetchall()

        daily_cursor = await db.execute(
            """SELECT DATE(timestamp) AS day,
                      SUM(cost_usd) AS daily_cost,
                      SUM(input_tokens + output_tokens) AS daily_tokens
               FROM token_log
               GROUP BY DATE(timestamp)
               ORDER BY day DESC
               LIMIT 30"""
        )
        daily_rows = await daily_cursor.fetchall()

    return {
        "by_model": [dict(r) for r in rows],
        "daily": [dict(r) for r in daily_rows],
        "total_usd": sum(r["total_cost"] or 0 for r in rows),
    }
