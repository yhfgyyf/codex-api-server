import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator

import httpx

from auth import TokenManager
from config import CODEX_BASE_URL, CODEX_MODELS, CODEX_RESPONSES_PATH, DEFAULT_CODEX_MODEL, MODEL_ALIASES
from db import save_log

logger = logging.getLogger("codex-api-server.proxy")


def _codex_url() -> str:
    base = CODEX_BASE_URL.rstrip("/")
    return f"{base}{CODEX_RESPONSES_PATH}"


def _resolve_model(model: str) -> str:
    """Map any model name to a valid Codex model.

    Priority: exact Codex model > explicit alias > prefix match > default.
    """
    if model in CODEX_MODELS:
        return model
    if model in MODEL_ALIASES:
        return MODEL_ALIASES[model]
    # Prefix match for versioned names (e.g. claude-sonnet-4-20250514 → claude-sonnet-4)
    for alias, target in MODEL_ALIASES.items():
        if model.startswith(alias):
            return target
    logger.info("Model '%s' not recognized, using default '%s'", model, DEFAULT_CODEX_MODEL)
    return DEFAULT_CODEX_MODEL


def _convert_tools_for_codex(tools: list) -> list:
    """Convert OpenAI Chat Completions tools format to Codex Responses format."""
    codex_tools = []
    for tool in tools:
        if tool.get("type") == "function":
            fn = tool.get("function", {})
            codex_tools.append({
                "type": "function",
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
                "strict": None,
            })
        else:
            codex_tools.append(tool)
    return codex_tools


def _convert_tool_calls_message(msg: dict) -> list[dict]:
    """Convert assistant message with tool_calls to Codex input items.

    Returns a list of top-level input items: a message (if text present)
    followed by separate function_call items (NOT nested in message content).
    """
    result = []
    content = msg.get("content")
    if content:
        result.append({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": content}],
        })
    for tc in msg.get("tool_calls", []):
        fn = tc.get("function", {})
        result.append({
            "type": "function_call",
            "call_id": tc.get("id", ""),
            "name": fn.get("name", ""),
            "arguments": fn.get("arguments", "{}"),
        })
    return result


def _clamp_reasoning_effort(effort: str, model: str) -> str:
    """Clamp reasoning effort to valid range per model."""
    m = model.lower()
    if "5.2" in m or "5.3" in m or "5.4" in m:
        if effort == "minimal":
            return "low"
    if "5.1" in m and "mini" not in m:
        if effort == "xhigh":
            return "high"
    return effort


def _chat_to_responses(body: dict) -> dict:
    """Convert OpenAI Chat Completions request to Codex Responses format.

    Accepts all standard OpenAI/vLLM Chat Completions parameters.
    Supported params are mapped to Codex Responses format.
    Unsupported params (temperature, top_p, max_tokens, etc.) are silently ignored.
    """
    messages = body.get("messages", [])
    model = _resolve_model(body.get("model", "gpt-5.4"))

    system_parts = []
    input_items = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        # Handle tool_calls in assistant messages
        if role == "assistant" and msg.get("tool_calls"):
            input_items.extend(_convert_tool_calls_message(msg))
            continue

        # Handle tool/function result messages
        if role == "tool":
            input_items.append({
                "type": "function_call_output",
                "call_id": msg.get("tool_call_id", ""),
                "output": content if isinstance(content, str) else json.dumps(content),
            })
            continue

        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") in ("text", "input_text"):
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
        "include": ["reasoning.encrypted_content"],
    }

    result["instructions"] = "\n\n".join(system_parts) if system_parts else "You are a helpful assistant."

    # -- Tools / function calling --
    tools = body.get("tools")
    if tools:
        result["tools"] = _convert_tools_for_codex(tools)
        result["tool_choice"] = body.get("tool_choice", "auto")
        result["parallel_tool_calls"] = body.get("parallel_tool_calls", True)
    else:
        result["tool_choice"] = "auto"
        result["parallel_tool_calls"] = True

    # -- Reasoning (always enabled so models output thinking process) --
    reasoning_effort = body.get("reasoning_effort")
    reasoning = body.get("reasoning")
    if reasoning_effort:
        effort = _clamp_reasoning_effort(str(reasoning_effort), model)
        result["reasoning"] = {"effort": effort, "summary": "auto"}
    elif isinstance(reasoning, dict):
        effort = reasoning.get("effort", "medium")
        effort = _clamp_reasoning_effort(str(effort), model)
        summary = reasoning.get("summary", "auto")
        result["reasoning"] = {"effort": effort, "summary": summary}
    else:
        # Default: enable reasoning with medium effort + auto summary
        result["reasoning"] = {"effort": "medium", "summary": "auto"}

    # -- All other OpenAI/vLLM params are silently ignored --
    # Ignored: temperature, top_p, top_k, n, max_tokens, max_completion_tokens,
    #   stop, presence_penalty, frequency_penalty, repetition_penalty,
    #   logit_bias, logprobs, top_logprobs, seed, response_format, user,
    #   stream_options, service_tier, min_p, best_of, echo, suffix,
    #   guided_json, guided_regex, guided_choice, guided_grammar,
    #   length_penalty, early_stopping, etc.

    ignored = [k for k in body if k not in {
        "model", "messages", "stream", "tools", "tool_choice", "parallel_tool_calls",
        "reasoning_effort", "reasoning", "instructions",
    }]
    if ignored:
        logger.debug("Ignored unsupported Chat Completions params: %s", ", ".join(ignored))

    return result


# ---------------------------------------------------------------------------
# Anthropic Messages API → Codex Responses conversion
# ---------------------------------------------------------------------------

def _anthropic_convert_tools(tools: list) -> list:
    """Convert Anthropic tool definitions to Codex Responses format."""
    codex_tools = []
    for tool in tools:
        codex_tools.append({
            "type": "function",
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {}),
            "strict": None,
        })
    return codex_tools


def _anthropic_to_responses(body: dict) -> dict:
    """Convert Anthropic Messages API request to Codex Responses format."""
    model = _resolve_model(body.get("model", "gpt-5.4"))
    messages = body.get("messages", [])
    system = body.get("system", "")

    input_items = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        # Handle tool_result blocks
        if role == "user" and isinstance(content, list):
            has_tool_result = any(
                isinstance(b, dict) and b.get("type") == "tool_result"
                for b in content
            )
            if has_tool_result:
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "tool_result":
                        output = block.get("content", "")
                        if isinstance(output, list):
                            output = "\n".join(
                                b.get("text", "") for b in output
                                if isinstance(b, dict) and b.get("type") == "text"
                            )
                        input_items.append({
                            "type": "function_call_output",
                            "call_id": block.get("tool_use_id", ""),
                            "output": output if isinstance(output, str) else json.dumps(output),
                        })
                    elif block.get("type") == "text":
                        input_items.append({
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": block.get("text", "")}],
                        })
                continue

        # Extract text from content (string or array of blocks)
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            text = "\n".join(text_parts)
        else:
            text = str(content)

        if role == "user":
            input_items.append({
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            })
        elif role == "assistant":
            # Check for tool_use blocks — function_call must be top-level input items
            if isinstance(content, list):
                text_parts = []
                fc_items = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text" and block.get("text"):
                        text_parts.append(block["text"])
                    elif block.get("type") == "tool_use":
                        inp = block.get("input", {})
                        fc_items.append({
                            "type": "function_call",
                            "call_id": block.get("id", ""),
                            "name": block.get("name", ""),
                            "arguments": json.dumps(inp) if isinstance(inp, dict) else str(inp),
                        })
                if text_parts or fc_items:
                    if text_parts:
                        input_items.append({
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "\n".join(text_parts)}],
                        })
                    input_items.extend(fc_items)
                    continue
            input_items.append({
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            })

    # Handle system prompt (string or array of blocks)
    if isinstance(system, list):
        system = "\n\n".join(
            b.get("text", "") for b in system
            if isinstance(b, dict) and b.get("type") == "text"
        )

    result = {
        "model": model,
        "store": False,
        "stream": True,
        "input": input_items,
        "instructions": system or "You are a helpful assistant.",
        "include": ["reasoning.encrypted_content"],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
    }

    # Tools
    tools = body.get("tools")
    if tools:
        result["tools"] = _anthropic_convert_tools(tools)
        # Map Anthropic tool_choice to Codex format
        tc = body.get("tool_choice")
        if isinstance(tc, dict):
            tc_type = tc.get("type", "auto")
            if tc_type == "any":
                result["tool_choice"] = "required"
            elif tc_type == "tool":
                result["tool_choice"] = {"type": "function", "name": tc.get("name", "")}
            else:
                result["tool_choice"] = tc_type
        elif isinstance(tc, str):
            result["tool_choice"] = tc

    # Reasoning — map Anthropic's thinking.budget_tokens to Codex reasoning
    thinking = body.get("thinking")
    if isinstance(thinking, dict) and thinking.get("type") == "enabled":
        budget = thinking.get("budget_tokens", 10000)
        if budget <= 2000:
            effort = "low"
        elif budget <= 8000:
            effort = "medium"
        else:
            effort = "high"
        result["reasoning"] = {"effort": effort, "summary": "auto"}
    else:
        result["reasoning"] = {"effort": "high", "summary": "auto"}

    return result


def _anthropic_usage(codex_usage: dict) -> dict:
    """Convert internal usage dict to Anthropic format."""
    return {
        "input_tokens": codex_usage.get("prompt_tokens", 0),
        "output_tokens": codex_usage.get("completion_tokens", 0),
    }


# Parameters that the Codex Responses API actually accepts
_CODEX_RESPONSES_ALLOWED_KEYS = {
    "model", "store", "stream", "instructions", "input", "include",
    "tool_choice", "parallel_tool_calls", "tools", "reasoning", "text",
    "prompt_cache_key",
}


def _sanitize_responses_body(body: dict) -> dict:
    """Remove unsupported parameters from a Responses API body."""
    sanitized = {k: v for k, v in body.items() if k in _CODEX_RESPONSES_ALLOWED_KEYS}
    removed = [k for k in body if k not in _CODEX_RESPONSES_ALLOWED_KEYS]
    if removed:
        logger.debug("Removed unsupported Responses params: %s", ", ".join(removed))
    return sanitized


def _extract_text_from_responses_event(event: dict) -> str | None:
    event_type = event.get("type", "")
    if event_type == "response.output_text.delta":
        return event.get("delta", "")
    return None


def _extract_tool_calls_from_response(response_obj: dict) -> list:
    """Extract function calls from Codex response, convert to OpenAI tool_calls format."""
    output = response_obj.get("output", [])
    tool_calls = []
    idx = 0
    for item in output:
        if item.get("type") == "function_call":
            tool_calls.append({
                "index": idx,
                "id": item.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                "type": "function",
                "function": {
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments", "{}"),
                },
            })
            idx += 1
    return tool_calls


def _extract_final_text_from_response(response_obj: dict) -> str:
    output = response_obj.get("output", [])
    text_parts = []
    for item in output:
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") in ("output_text", "text"):
                    text_parts.append(part.get("text", ""))
    return "".join(text_parts)


def _extract_reasoning_from_response(response_obj: dict) -> str:
    """Extract reasoning/thinking summary text from Codex response output."""
    output = response_obj.get("output", [])
    reasoning_parts = []
    for item in output:
        if item.get("type") == "reasoning":
            for part in item.get("summary", []):
                if part.get("type") == "summary_text":
                    reasoning_parts.append(part.get("text", ""))
    return "".join(reasoning_parts)


def _extract_usage_from_response(response_obj: dict) -> dict:
    usage = response_obj.get("usage", {})
    return {
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("total_tokens", usage.get("input_tokens", 0) + usage.get("output_tokens", 0)),
    }


async def _stream_and_collect(client: httpx.AsyncClient, url: str, body: dict, headers: dict):
    """Stream from Codex, yield parsed events. Retry once on 401/403."""
    async with client.stream("POST", url, json=body, headers=headers) as resp:
        if resp.status_code not in (401, 403):
            if resp.status_code >= 400:
                error_body = await resp.aread()
                yield "error", error_body
                return
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    yield "event", json.loads(data_str)
                except json.JSONDecodeError:
                    continue
            return
    # 401/403 path: caller should refresh and call again
    yield "auth_failed", None


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

    async def _stream_with_retry(self, url: str, body: dict):
        """Stream from Codex with one retry on auth failure."""
        headers = await self.token_manager.get_headers()
        auth_failed = False
        async for kind, data in _stream_and_collect(self.client, url, body, headers):
            if kind == "auth_failed":
                auth_failed = True
                break
            yield kind, data
        if not auth_failed:
            return

        # Retry after refresh
        await self.token_manager.force_refresh()
        headers = await self.token_manager.get_headers()
        async for kind, data in _stream_and_collect(self.client, url, body, headers):
            if kind == "auth_failed":
                yield "error", b'{"error":{"message":"Authentication failed after retry","type":"auth_error"}}'
                return
            yield kind, data

    async def chat_completions(self, body: dict) -> dict:
        """Non-streaming chat completion via Codex Responses API."""
        t0 = time.monotonic()
        codex_body = _chat_to_responses(body)
        codex_body["stream"] = True

        model = codex_body["model"]
        run_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        url = _codex_url()

        full_text = ""
        reasoning_text = ""
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        final_response_obj = None

        async for kind, data in self._stream_with_retry(url, codex_body):
            if kind == "error":
                try:
                    error_data = json.loads(data)
                except Exception:
                    error_data = {"error": {"message": data.decode(errors="replace"), "type": "api_error"}}
                duration_ms = int((time.monotonic() - t0) * 1000)
                await save_log(run_id, "chat/completions", model, False, body, error_data,
                               status="error", duration_ms=duration_ms)
                return error_data

            event = data
            event_type = event.get("type", "")
            delta = _extract_text_from_responses_event(event)
            if delta:
                full_text += delta
            if event_type == "response.reasoning_summary_text.delta":
                reasoning_text += event.get("delta", "")
            if event_type in ("response.completed", "response.done"):
                final_response_obj = event.get("response", {})
                if not full_text:
                    full_text = _extract_final_text_from_response(final_response_obj)
                if not reasoning_text:
                    reasoning_text = _extract_reasoning_from_response(final_response_obj)
                usage = _extract_usage_from_response(final_response_obj)

        # Build response message
        message: dict = {"role": "assistant", "content": full_text or None}
        if reasoning_text:
            message["reasoning_content"] = reasoning_text
        finish_reason = "stop"
        if final_response_obj:
            tool_calls = _extract_tool_calls_from_response(final_response_obj)
            if tool_calls:
                message["tool_calls"] = tool_calls
                finish_reason = "tool_calls"
                if not full_text:
                    message["content"] = None

        result = {
            "id": run_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }],
            "usage": usage,
        }

        duration_ms = int((time.monotonic() - t0) * 1000)
        await save_log(
            run_id, "chat/completions", model, False, body, result,
            status="ok",
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            duration_ms=duration_ms,
        )
        return result

    async def chat_completions_stream(self, body: dict) -> AsyncGenerator[bytes, None]:
        """Streaming chat completion via Codex Responses API, converting to OpenAI SSE format."""
        t0 = time.monotonic()
        codex_body = _chat_to_responses(body)
        codex_body["stream"] = True

        model = codex_body["model"]
        run_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        url = _codex_url()
        headers = await self.token_manager.get_headers()

        sent_role = False
        collected_text = ""
        collected_reasoning = ""
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        status = "ok"
        has_tool_calls = False
        tool_call_index = 0

        async with self.client.stream("POST", url, json=codex_body, headers=headers) as resp:
            if resp.status_code >= 400:
                error_body = await resp.aread()
                status = "error"
                duration_ms = int((time.monotonic() - t0) * 1000)
                try:
                    err = json.loads(error_body)
                except Exception:
                    err = {"error": {"message": error_body.decode(errors="replace")}}
                await save_log(run_id, "chat/completions", model, True, body, err,
                               status="error", duration_ms=duration_ms)
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

                if not sent_role and event_type in (
                    "response.output_text.delta",
                    "response.output_item.added",
                    "response.created",
                    "response.in_progress",
                    "response.function_call_arguments.delta",
                    "response.reasoning_summary_text.delta",
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

                # Reasoning/thinking delta → reasoning_content (DeepSeek/vLLM format)
                if event_type == "response.reasoning_summary_text.delta":
                    r_delta = event.get("delta", "")
                    collected_reasoning += r_delta
                    chunk = {
                        "id": run_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [{"index": 0, "delta": {"reasoning_content": r_delta}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n".encode()

                # Text content delta
                delta = _extract_text_from_responses_event(event)
                if delta is not None:
                    collected_text += delta
                    chunk = {
                        "id": run_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n".encode()

                # Function call streaming — emit tool_calls chunks
                if event_type == "response.output_item.added" and event.get("item", {}).get("type") == "function_call":
                    has_tool_calls = True
                    item = event["item"]
                    tc = {
                        "index": tool_call_index,
                        "id": item.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                        "type": "function",
                        "function": {"name": item.get("name", ""), "arguments": ""},
                    }
                    chunk = {
                        "id": run_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [{"index": 0, "delta": {"tool_calls": [tc]}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n".encode()

                if event_type == "response.function_call_arguments.delta":
                    args_delta = event.get("delta", "")
                    tc = {
                        "index": tool_call_index,
                        "function": {"arguments": args_delta},
                    }
                    chunk = {
                        "id": run_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [{"index": 0, "delta": {"tool_calls": [tc]}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n".encode()

                if event_type == "response.function_call_arguments.done":
                    tool_call_index += 1

                if event_type in ("response.completed", "response.done"):
                    response_obj = event.get("response", {})
                    usage = _extract_usage_from_response(response_obj)
                    finish = "tool_calls" if has_tool_calls else "stop"
                    chunk = {
                        "id": run_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                        "usage": usage,
                    }
                    yield f"data: {json.dumps(chunk)}\n\n".encode()

                if event_type in ("error", "response.failed"):
                    status = "error"
                    error_msg = event.get("message", event.get("response", {}).get("error", {}).get("message", "Unknown error"))
                    error_chunk = {"error": {"message": error_msg, "type": "api_error"}}
                    yield f"data: {json.dumps(error_chunk)}\n\n".encode()

        yield b"data: [DONE]\n\n"

        duration_ms = int((time.monotonic() - t0) * 1000)
        response_summary = {
            "id": run_id,
            "object": "chat.completion",
            "model": model,
            "content": collected_text,
            "reasoning_content": collected_reasoning or None,
            "usage": usage,
        }
        await save_log(
            run_id, "chat/completions", model, True, body, response_summary,
            status=status,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            duration_ms=duration_ms,
        )

    async def responses_proxy(self, body: dict) -> dict:
        """Direct proxy to Codex Responses API (non-streaming)."""
        t0 = time.monotonic()
        model = _resolve_model(body.get("model", "gpt-5.4"))
        body_copy = _sanitize_responses_body({**body, "model": model, "stream": True, "store": False})
        request_id = f"resp-{uuid.uuid4().hex[:24]}"
        url = _codex_url()

        final_response = None
        async for kind, data in self._stream_with_retry(url, body_copy):
            if kind == "error":
                duration_ms = int((time.monotonic() - t0) * 1000)
                try:
                    err = json.loads(data)
                except Exception:
                    err = {"error": {"message": data.decode(errors="replace")}}
                await save_log(request_id, "responses", model, False, body, err,
                               status="error", duration_ms=duration_ms)
                return err

            event = data
            if event.get("type") in ("response.completed", "response.done"):
                final_response = event.get("response", event)

        result = final_response or {"error": {"message": "No response received"}}
        duration_ms = int((time.monotonic() - t0) * 1000)

        usage = _extract_usage_from_response(result) if final_response else {}
        await save_log(
            request_id, "responses", model, False, body, result,
            status="ok" if final_response else "error",
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            duration_ms=duration_ms,
        )
        return result

    async def responses_proxy_stream(self, body: dict) -> AsyncGenerator[bytes, None]:
        """Direct proxy to Codex Responses API (streaming passthrough)."""
        t0 = time.monotonic()
        model = _resolve_model(body.get("model", "gpt-5.4"))
        body_copy = _sanitize_responses_body({**body, "model": model, "stream": True, "store": False})
        request_id = f"resp-{uuid.uuid4().hex[:24]}"
        url = _codex_url()
        headers = await self.token_manager.get_headers()

        collected_text = ""
        usage = {}
        status = "ok"

        async with self.client.stream("POST", url, json=body_copy, headers=headers) as resp:
            if resp.status_code >= 400:
                error_body = await resp.aread()
                duration_ms = int((time.monotonic() - t0) * 1000)
                try:
                    err = json.loads(error_body)
                except Exception:
                    err = {"error": {"message": error_body.decode(errors="replace")}}
                await save_log(request_id, "responses", model, True, body, err,
                               status="error", duration_ms=duration_ms)
                yield b"data: " + error_body + b"\n\n"
                yield b"data: [DONE]\n\n"
                return

            async for line in resp.aiter_lines():
                if line.startswith("data:"):
                    data_str = line[5:].strip()
                    if data_str != "[DONE]":
                        try:
                            event = json.loads(data_str)
                            delta = _extract_text_from_responses_event(event)
                            if delta:
                                collected_text += delta
                            event_type = event.get("type", "")
                            if event_type in ("response.completed", "response.done"):
                                response_obj = event.get("response", {})
                                usage = _extract_usage_from_response(response_obj)
                                if not collected_text:
                                    collected_text = _extract_final_text_from_response(response_obj)
                            if event_type in ("error", "response.failed"):
                                status = "error"
                        except json.JSONDecodeError:
                            pass
                yield (line + "\n").encode()
            yield b"\n"

        duration_ms = int((time.monotonic() - t0) * 1000)
        response_summary = {
            "id": request_id,
            "model": model,
            "content": collected_text,
            "usage": usage,
        }
        await save_log(
            request_id, "responses", model, True, body, response_summary,
            status=status,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            duration_ms=duration_ms,
        )

    # -------------------------------------------------------------------
    # Anthropic Messages API
    # -------------------------------------------------------------------

    async def anthropic_messages(self, body: dict) -> dict:
        """Non-streaming Anthropic Messages API via Codex Responses."""
        t0 = time.monotonic()
        codex_body = _anthropic_to_responses(body)
        model = codex_body["model"]
        msg_id = f"msg_{uuid.uuid4().hex[:24]}"
        url = _codex_url()

        full_text = ""
        reasoning_text = ""
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        final_response_obj = None
        tool_calls: list[dict] = []

        async for kind, data in self._stream_with_retry(url, codex_body):
            if kind == "error":
                try:
                    error_data = json.loads(data)
                except Exception:
                    error_data = {"error": {"message": data.decode(errors="replace"), "type": "api_error"}}
                duration_ms = int((time.monotonic() - t0) * 1000)
                err_resp = {
                    "type": "error",
                    "error": {
                        "type": error_data.get("error", {}).get("type", "api_error"),
                        "message": error_data.get("error", {}).get("message", str(error_data)),
                    },
                }
                await save_log(msg_id, "messages", model, False, body, err_resp,
                               status="error", duration_ms=duration_ms)
                return err_resp

            event = data
            event_type = event.get("type", "")
            delta = _extract_text_from_responses_event(event)
            if delta:
                full_text += delta
            if event_type == "response.reasoning_summary_text.delta":
                reasoning_text += event.get("delta", "")
            if event_type in ("response.completed", "response.done"):
                final_response_obj = event.get("response", {})
                if not full_text:
                    full_text = _extract_final_text_from_response(final_response_obj)
                if not reasoning_text:
                    reasoning_text = _extract_reasoning_from_response(final_response_obj)
                usage = _extract_usage_from_response(final_response_obj)
                tool_calls = _extract_tool_calls_from_response(final_response_obj)

        # Build Anthropic content blocks
        content_blocks: list[dict] = []
        if reasoning_text:
            content_blocks.append({"type": "thinking", "thinking": reasoning_text})
        if full_text:
            content_blocks.append({"type": "text", "text": full_text})
        for tc in tool_calls:
            fn = tc.get("function", {})
            try:
                inp = json.loads(fn.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                inp = {}
            content_blocks.append({
                "type": "tool_use",
                "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                "name": fn.get("name", ""),
                "input": inp,
            })

        stop_reason = "tool_use" if tool_calls else "end_turn"
        if not content_blocks:
            content_blocks.append({"type": "text", "text": ""})

        result = {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content_blocks,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": _anthropic_usage(usage),
        }

        duration_ms = int((time.monotonic() - t0) * 1000)
        await save_log(
            msg_id, "messages", model, False, body, result,
            status="ok",
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            duration_ms=duration_ms,
        )
        return result

    async def anthropic_messages_stream(self, body: dict) -> AsyncGenerator[bytes, None]:
        """Streaming Anthropic Messages API via Codex Responses."""
        t0 = time.monotonic()
        codex_body = _anthropic_to_responses(body)
        model = codex_body["model"]
        msg_id = f"msg_{uuid.uuid4().hex[:24]}"
        url = _codex_url()
        headers = await self.token_manager.get_headers()

        collected_text = ""
        collected_reasoning = ""
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        status = "ok"
        block_index = 0
        in_thinking = False
        in_text = False
        has_tool_calls = False
        tool_blocks: list[dict] = []

        def _sse(event_type: str, data: dict) -> bytes:
            return f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode()

        async with self.client.stream("POST", url, json=codex_body, headers=headers) as resp:
            if resp.status_code >= 400:
                error_body = await resp.aread()
                status = "error"
                duration_ms = int((time.monotonic() - t0) * 1000)
                try:
                    err = json.loads(error_body)
                except Exception:
                    err = {"error": {"message": error_body.decode(errors="replace")}}
                await save_log(msg_id, "messages", model, True, body, err,
                               status="error", duration_ms=duration_ms)
                yield _sse("error", {
                    "type": "error",
                    "error": {"type": "api_error", "message": err.get("error", {}).get("message", str(err))},
                })
                return

            # Emit message_start
            yield _sse("message_start", {
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            })

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

                # Reasoning/thinking delta
                if event_type == "response.reasoning_summary_text.delta":
                    r_delta = event.get("delta", "")
                    collected_reasoning += r_delta
                    if not in_thinking:
                        in_thinking = True
                        yield _sse("content_block_start", {
                            "type": "content_block_start",
                            "index": block_index,
                            "content_block": {"type": "thinking", "thinking": ""},
                        })
                    yield _sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {"type": "thinking_delta", "thinking": r_delta},
                    })

                # Close thinking block when text starts
                if event_type == "response.output_text.delta" and in_thinking and not in_text:
                    yield _sse("content_block_stop", {
                        "type": "content_block_stop",
                        "index": block_index,
                    })
                    block_index += 1
                    in_thinking = False

                # Text content delta
                text_delta = _extract_text_from_responses_event(event)
                if text_delta is not None:
                    collected_text += text_delta
                    if not in_text:
                        in_text = True
                        yield _sse("content_block_start", {
                            "type": "content_block_start",
                            "index": block_index,
                            "content_block": {"type": "text", "text": ""},
                        })
                    yield _sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {"type": "text_delta", "text": text_delta},
                    })

                # Function call tracking
                if event_type == "response.output_item.added" and event.get("item", {}).get("type") == "function_call":
                    has_tool_calls = True
                    item = event["item"]
                    tool_blocks.append({
                        "id": item.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                        "name": item.get("name", ""),
                        "arguments": "",
                    })

                if event_type == "response.function_call_arguments.delta":
                    if tool_blocks:
                        tool_blocks[-1]["arguments"] += event.get("delta", "")

                if event_type == "response.function_call_arguments.done":
                    if tool_blocks:
                        tb = tool_blocks[-1]
                        if in_text:
                            yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
                            block_index += 1
                            in_text = False
                        if in_thinking:
                            yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
                            block_index += 1
                            in_thinking = False
                        yield _sse("content_block_start", {
                            "type": "content_block_start",
                            "index": block_index,
                            "content_block": {
                                "type": "tool_use",
                                "id": tb["id"],
                                "name": tb["name"],
                                "input": {},
                            },
                        })
                        yield _sse("content_block_delta", {
                            "type": "content_block_delta",
                            "index": block_index,
                            "delta": {"type": "input_json_delta", "partial_json": tb["arguments"]},
                        })
                        yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
                        block_index += 1

                if event_type in ("response.completed", "response.done"):
                    response_obj = event.get("response", {})
                    usage = _extract_usage_from_response(response_obj)

                if event_type in ("error", "response.failed"):
                    status = "error"

            # Close any open blocks
            if in_thinking:
                yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
                block_index += 1
            if in_text:
                yield _sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
                block_index += 1

        # Emit message_delta + message_stop
        stop_reason = "tool_use" if has_tool_calls else "end_turn"
        yield _sse("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": usage.get("completion_tokens", 0)},
        })
        yield _sse("message_stop", {"type": "message_stop"})
