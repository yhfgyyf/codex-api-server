import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator

import httpx

from auth import TokenManager
from config import CODEX_BASE_URL, CODEX_RESPONSES_PATH

logger = logging.getLogger("codex-api-server.proxy")


def _codex_url() -> str:
    base = CODEX_BASE_URL.rstrip("/")
    return f"{base}{CODEX_RESPONSES_PATH}"


def _chat_to_responses(body: dict) -> dict:
    """Convert OpenAI Chat Completions request to Codex Responses format."""
    messages = body.get("messages", [])
    model = body.get("model", "gpt-5.4")

    # Separate system messages and conversation
    system_parts = []
    input_items = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        # Handle content that's a list of parts
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif part.get("type") == "input_text":
                        text_parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    text_parts.append(part)
            content = "\n".join(text_parts)

        if role in ("system", "developer"):
            system_parts.append(content)
        elif role == "user":
            input_items.append({
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": content}],
            })
        elif role == "assistant":
            input_items.append({
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": content}],
            })

    result = {
        "model": model,
        "store": False,
        "stream": body.get("stream", False),
        "input": input_items,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "include": ["reasoning.encrypted_content"],
    }

    # Codex requires instructions field
    result["instructions"] = "\n\n".join(system_parts) if system_parts else "You are a helpful assistant."

    # Map temperature
    if "temperature" in body:
        result["temperature"] = body["temperature"]

    # Note: Codex doesn't support max_output_tokens; max_tokens is ignored

    # Map reasoning_effort
    if "reasoning_effort" in body:
        result["reasoning"] = {"effort": body["reasoning_effort"]}

    return result


def _extract_text_from_responses_event(event: dict) -> str | None:
    """Extract text delta from a Codex Responses SSE event."""
    event_type = event.get("type", "")
    if event_type == "response.output_text.delta":
        return event.get("delta", "")
    return None


def _extract_final_text_from_response(response_obj: dict) -> str:
    """Extract full text from a completed response object."""
    output = response_obj.get("output", [])
    text_parts = []
    for item in output:
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") in ("output_text", "text"):
                    text_parts.append(part.get("text", ""))
    return "".join(text_parts)


def _extract_usage_from_response(response_obj: dict) -> dict:
    """Extract usage from response object."""
    usage = response_obj.get("usage", {})
    return {
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("total_tokens", usage.get("input_tokens", 0) + usage.get("output_tokens", 0)),
    }


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

    async def _request_with_retry(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Make request, retry once on auth failure."""
        headers = await self.token_manager.get_headers()
        headers.update(kwargs.pop("extra_headers", {}))
        resp = await self.client.request(method, url, headers=headers, **kwargs)
        if resp.status_code in (401, 403):
            logger.warning("Got %d, refreshing token...", resp.status_code)
            await self.token_manager._do_refresh()
            headers = await self.token_manager.get_headers()
            resp = await self.client.request(method, url, headers=headers, **kwargs)
        return resp

    async def chat_completions(self, body: dict) -> dict:
        """Non-streaming chat completion via Codex Responses API."""
        codex_body = _chat_to_responses(body)
        codex_body["stream"] = True  # always stream from Codex, collect full response

        model = body.get("model", "gpt-5.4")
        run_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        url = _codex_url()

        headers = await self.token_manager.get_headers()
        full_text = ""
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        async with self.client.stream("POST", url, json=codex_body, headers=headers) as resp:
            if resp.status_code in (401, 403):
                await self.token_manager._do_refresh()
                headers = await self.token_manager.get_headers()
            elif resp.status_code >= 400:
                error_body = await resp.aread()
                try:
                    error_data = json.loads(error_body)
                except Exception:
                    error_data = {"error": {"message": error_body.decode(errors="replace"), "type": "api_error"}}
                return error_data

            if resp.status_code < 400:
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        event = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    delta = _extract_text_from_responses_event(event)
                    if delta:
                        full_text += delta

                    event_type = event.get("type", "")
                    if event_type in ("response.completed", "response.done"):
                        response_obj = event.get("response", {})
                        if not full_text:
                            full_text = _extract_final_text_from_response(response_obj)
                        usage = _extract_usage_from_response(response_obj)

                return {
                    "id": run_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": full_text},
                        "finish_reason": "stop",
                    }],
                    "usage": usage,
                }

        # Retry after token refresh
        headers = await self.token_manager.get_headers()
        async with self.client.stream("POST", url, json=codex_body, headers=headers) as resp:
            if resp.status_code >= 400:
                error_body = await resp.aread()
                try:
                    return json.loads(error_body)
                except Exception:
                    return {"error": {"message": error_body.decode(errors="replace"), "type": "api_error"}}

            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                delta = _extract_text_from_responses_event(event)
                if delta:
                    full_text += delta
                event_type = event.get("type", "")
                if event_type in ("response.completed", "response.done"):
                    response_obj = event.get("response", {})
                    if not full_text:
                        full_text = _extract_final_text_from_response(response_obj)
                    usage = _extract_usage_from_response(response_obj)

        return {
            "id": run_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": full_text},
                "finish_reason": "stop",
            }],
            "usage": usage,
        }

    async def chat_completions_stream(self, body: dict) -> AsyncGenerator[bytes, None]:
        """Streaming chat completion via Codex Responses API, converting to OpenAI SSE format."""
        codex_body = _chat_to_responses(body)
        codex_body["stream"] = True

        model = body.get("model", "gpt-5.4")
        run_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        url = _codex_url()
        headers = await self.token_manager.get_headers()

        sent_role = False

        async with self.client.stream("POST", url, json=codex_body, headers=headers) as resp:
            if resp.status_code >= 400:
                error_body = await resp.aread()
                yield b"data: " + error_body + b"\n\n"
                yield b"data: [DONE]\n\n"
                return

            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type", "")

                # Send initial role chunk
                if not sent_role and event_type in (
                    "response.output_text.delta",
                    "response.output_item.added",
                    "response.created",
                    "response.in_progress",
                ):
                    sent_role = True
                    chunk = {
                        "id": run_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n".encode()

                # Text deltas
                delta = _extract_text_from_responses_event(event)
                if delta is not None:
                    chunk = {
                        "id": run_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n".encode()

                # Completion
                if event_type in ("response.completed", "response.done"):
                    response_obj = event.get("response", {})
                    usage = _extract_usage_from_response(response_obj)
                    chunk = {
                        "id": run_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        "usage": usage,
                    }
                    yield f"data: {json.dumps(chunk)}\n\n".encode()

                # Error events
                if event_type == "error" or event_type == "response.failed":
                    error_msg = event.get("message", event.get("response", {}).get("error", {}).get("message", "Unknown error"))
                    error_chunk = {"error": {"message": error_msg, "type": "api_error"}}
                    yield f"data: {json.dumps(error_chunk)}\n\n".encode()

        yield b"data: [DONE]\n\n"

    async def responses_proxy(self, body: dict) -> dict:
        """Direct proxy to Codex Responses API (non-streaming)."""
        body["stream"] = True  # Codex requires streaming
        body["store"] = False  # Codex requires store=false
        url = _codex_url()
        headers = await self.token_manager.get_headers()

        final_response = None
        async with self.client.stream("POST", url, json=body, headers=headers) as resp:
            if resp.status_code >= 400:
                error_body = await resp.aread()
                try:
                    return json.loads(error_body)
                except Exception:
                    return {"error": {"message": error_body.decode(errors="replace")}}

            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                if event.get("type") in ("response.completed", "response.done"):
                    final_response = event.get("response", event)

        return final_response or {"error": {"message": "No response received"}}

    async def responses_proxy_stream(self, body: dict) -> AsyncGenerator[bytes, None]:
        """Direct proxy to Codex Responses API (streaming passthrough)."""
        body["stream"] = True
        body["store"] = False
        url = _codex_url()
        headers = await self.token_manager.get_headers()

        async with self.client.stream("POST", url, json=body, headers=headers) as resp:
            if resp.status_code >= 400:
                error_body = await resp.aread()
                yield b"data: " + error_body + b"\n\n"
                yield b"data: [DONE]\n\n"
                return

            async for chunk in resp.aiter_bytes():
                yield chunk
