"""
data/lumenx_client.py
Async HTTP client for the LumenX REST API.
All calls are retried with exponential backoff.
"""
from __future__ import annotations

import logging
import ssl
from typing import Any

import httpx
import truststore
from tenacity import retry, stop_after_attempt, wait_exponential

from data.models import Message, Product, Thread

logger = logging.getLogger(__name__)

# Use the OS-native certificate store (handles Windows corporate certs, macOS keychain).
_SSL_CONTEXT = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)


class LumenXClient:
    """Thin async wrapper around the LumenX admin + public REST API."""

    def __init__(self, base_url: str, admin_token: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._headers = {
            "X-Admin-Token": admin_token,
            "Content-Type": "application/json",
        }
        self._client: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> LumenXClient:
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=self._headers,
            timeout=30.0,
            verify=_SSL_CONTEXT,
        )
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._client:
            await self._client.aclose()

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("Use LumenXClient as an async context manager.")
        return self._client

    # ------------------------------------------------------------------
    # Admin endpoints
    # ------------------------------------------------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def get_stats(self) -> dict[str, Any]:
        """GET /api/admin/stats"""
        client = self._ensure_client()
        resp = await client.get("/api/admin/stats")
        resp.raise_for_status()
        return resp.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def get_inbox(self, since: str | None = None) -> dict[str, Any]:
        """GET /api/admin/inbox — returns threads awaiting admin reply."""
        client = self._ensure_client()
        params = {"since": since} if since else {}
        resp = await client.get("/api/admin/inbox", params=params)
        resp.raise_for_status()
        return resp.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def get_threads(self) -> list[dict[str, Any]]:
        """GET /api/admin/threads"""
        client = self._ensure_client()
        resp = await client.get("/api/admin/threads")
        resp.raise_for_status()
        return resp.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def get_thread(self, thread_id: str) -> dict[str, Any]:
        """GET /api/admin/threads/{id} — full thread with all messages."""
        client = self._ensure_client()
        resp = await client.get(f"/api/admin/threads/{thread_id}")
        resp.raise_for_status()
        return resp.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def send_reply(
        self,
        thread_id: str,
        text: str,
        draft_source: str = "agent",
        confidence: float | None = None,
    ) -> dict[str, Any]:
        """POST /api/admin/threads/{id}/reply"""
        client = self._ensure_client()
        body: dict[str, Any] = {"text": text, "draft_source": draft_source}
        if confidence is not None:
            body["confidence"] = confidence
        resp = await client.post(f"/api/admin/threads/{thread_id}/reply", json=body)
        resp.raise_for_status()
        logger.info("Sent reply to thread %s (confidence=%.3f)", thread_id, confidence or 0)
        return resp.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def mark_read(self, thread_id: str) -> dict[str, Any]:
        """POST /api/admin/threads/{id}/mark-read"""
        client = self._ensure_client()
        resp = await client.post(f"/api/admin/threads/{thread_id}/mark-read")
        resp.raise_for_status()
        return resp.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def export_all(self) -> dict[str, Any]:
        """GET /api/admin/export — full dump for bootstrapping."""
        client = self._ensure_client()
        resp = await client.get("/api/admin/export")
        resp.raise_for_status()
        return resp.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def get_products(self) -> list[dict[str, Any]]:
        """
        GET /api/admin/products — returns the products array.
        The endpoint returns {count, company, products}; this method extracts the list.
        Use get_products_full() if you also need company-wide policy info.
        """
        full = await self.get_products_full()
        return full.get("products", [])

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def get_products_full(self) -> dict[str, Any]:
        """GET /api/admin/products — full response (count + company + products)."""
        client = self._ensure_client()
        resp = await client.get("/api/admin/products")
        resp.raise_for_status()
        return resp.json()

    async def get_company_info(self) -> dict[str, Any]:
        """Company-wide policies (refund_window_days, free_trial_days, etc.)."""
        full = await self.get_products_full()
        return full.get("company", {})

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def get_product(self, product_id: str) -> dict[str, Any]:
        """GET /api/admin/products/{id}"""
        client = self._ensure_client()
        resp = await client.get(f"/api/admin/products/{product_id}")
        resp.raise_for_status()
        return resp.json()
