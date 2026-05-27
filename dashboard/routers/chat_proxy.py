"""
dashboard/routers/chat_proxy.py

Same-origin proxy for the LumenX public customer-chat endpoints. The local
/chat page calls these so the browser is not blocked by CORS (LumenX does
not return Access-Control-Allow-Origin for cross-origin requests).

Routes mirror the LumenX public API one-for-one:
  GET  /api/chat/threads?username=X
  POST /api/chat/threads                       body: { username, display_name?, product_id?, intent? }
  GET  /api/chat/threads/{thread_id}           ?viewer=customer
  POST /api/chat/threads/{thread_id}/messages  body: { role, text }
"""
from __future__ import annotations

import os
import ssl
from typing import Any

import httpx
import truststore
from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/api/chat", tags=["chat-proxy"])

LUMENX_BASE = os.getenv("LUMENX_BASE_URL", "https://lumenx-demo.up.railway.app")

# Use the OS-native cert store (Windows corporate cert chains, etc.)
_SSL = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)


async def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=LUMENX_BASE, timeout=20.0, verify=_SSL)


@router.get("/threads")
async def list_threads(username: str | None = None) -> Any:
    params = {"username": username} if username else {}
    async with await _client() as c:
        resp = await c.get("/api/threads", params=params)
        if not resp.is_success:
            raise HTTPException(resp.status_code, resp.text)
        return resp.json()


@router.post("/threads")
async def create_thread(request: Request) -> Any:
    body = await request.json()
    async with await _client() as c:
        resp = await c.post("/api/threads", json=body)
        if not resp.is_success:
            raise HTTPException(resp.status_code, resp.text)
        return resp.json()


@router.get("/threads/{thread_id}")
async def get_thread(thread_id: str, viewer: str = "customer") -> Any:
    async with await _client() as c:
        resp = await c.get(f"/api/threads/{thread_id}", params={"viewer": viewer})
        if not resp.is_success:
            raise HTTPException(resp.status_code, resp.text)
        return resp.json()


@router.post("/threads/{thread_id}/messages")
async def post_message(thread_id: str, request: Request) -> Any:
    body = await request.json()
    async with await _client() as c:
        resp = await c.post(f"/api/threads/{thread_id}/messages", json=body)
        if not resp.is_success:
            raise HTTPException(resp.status_code, resp.text)
        return resp.json()
