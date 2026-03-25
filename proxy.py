import logging
from collections.abc import AsyncGenerator

import httpx

from auth import TokenManager
from config import UPSTREAM_BASE_URL

logger = logging.getLogger("codex-api-server.proxy")


class OpenAIProxy:
    def __init__(self, token_manager: TokenManager):
        self.token_manager = token_manager
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0),
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
            http2=True,
        )

    async def close(self):
        await self.client.aclose()

    def _build_url(self, path: str) -> str:
        base = UPSTREAM_BASE_URL.rstrip("/")
        return f"{base}{path}"

    async def proxy_request(self, method: str, path: str, body: dict | None = None) -> httpx.Response:
        """Non-streaming proxy request with auto-retry on 401/403."""
        headers = await self.token_manager.get_headers()
        headers["Content-Type"] = "application/json"
        url = self._build_url(path)

        resp = await self.client.request(
            method, url, json=body, headers=headers,
        )

        # Retry once on auth failure
        if resp.status_code in (401, 403):
            logger.warning("Got %d from upstream, refreshing token and retrying...", resp.status_code)
            await self.token_manager._do_refresh()
            headers = await self.token_manager.get_headers()
            headers["Content-Type"] = "application/json"
            resp = await self.client.request(
                method, url, json=body, headers=headers,
            )

        return resp

    async def proxy_stream(self, path: str, body: dict) -> AsyncGenerator[bytes, None]:
        """Streaming proxy request, yields raw SSE bytes."""
        headers = await self.token_manager.get_headers()
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "text/event-stream"
        url = self._build_url(path)

        async with self.client.stream(
            "POST", url, json=body, headers=headers,
        ) as resp:
            if resp.status_code in (401, 403):
                logger.warning("Got %d on stream, refreshing token and retrying...", resp.status_code)
                await self.token_manager._do_refresh()
                headers = await self.token_manager.get_headers()
                headers["Content-Type"] = "application/json"
                headers["Accept"] = "text/event-stream"

            if resp.status_code in (401, 403):
                # Re-stream after refresh
                pass
            else:
                async for chunk in resp.aiter_bytes():
                    yield chunk
                return

        # Retry stream after token refresh
        headers = await self.token_manager.get_headers()
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "text/event-stream"
        async with self.client.stream(
            "POST", url, json=body, headers=headers,
        ) as resp:
            if resp.status_code >= 400:
                error_body = await resp.aread()
                yield error_body
                return
            async for chunk in resp.aiter_bytes():
                yield chunk
