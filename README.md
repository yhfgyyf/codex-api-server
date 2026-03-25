# Codex API Server

OpenAI API-compatible proxy server that uses ChatGPT Plus (Codex) OAuth credentials to provide standard `/v1/chat/completions` and `/v1/responses` endpoints.

## Features

- **OpenAI API compatible** — works with any OpenAI SDK, vLLM client, or tool
- **Reasoning model support** — streams `reasoning_content` (thinking process) in DeepSeek/vLLM format
- **Function calling** — full tool/function calling support with streaming tool_calls
- **Auto token refresh** — reads `~/.codex/auth.json`, refreshes OAuth tokens before expiry
- **Streaming SSE** — full Server-Sent Events support for chat completions
- **Responses API** — proxy passthrough for OpenAI's `/v1/responses` endpoint
- **vLLM parameter compatible** — accepts all OpenAI/vLLM params, silently ignores unsupported ones
- **Request logging** — all API calls stored in SQLite with full request/response data
- **JSONL export** — download or export logs for analysis and fine-tuning
- **Async & connection pooling** — built on FastAPI + httpx for high concurrency

## Supported Models

- `gpt-5.4`
- `gpt-5.3-codex`
- `gpt-5.3-codex-spark`
- `gpt-5.2-codex`
- `gpt-5.1-codex`

## Prerequisites

- Python 3.11+
- A valid `~/.codex/auth.json` file with ChatGPT OAuth credentials (from [OpenAI Codex CLI](https://github.com/openai/codex))

## Installation

```bash
# Create conda environment
conda create -n codex-api-server python=3.11 -y
conda activate codex-api-server

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
conda activate codex-api-server
python server.py
```

The server starts on `http://127.0.0.1:18888` by default.

## Usage

### With curl

```bash
# List models
curl http://localhost:18888/v1/models

# Chat completion
curl http://localhost:18888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-5.4","messages":[{"role":"user","content":"Hello"}]}'

# Streaming chat completion
curl -N http://localhost:18888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-5.4","messages":[{"role":"user","content":"Hello"}],"stream":true}'

# With reasoning effort control
curl http://localhost:18888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-5.4","messages":[{"role":"user","content":"Solve this step by step: 15*37"}],"reasoning_effort":"high"}'
```

### With OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:18888/v1", api_key="any")

# List models
models = client.models.list()

# Chat completion (with reasoning output)
resp = client.chat.completions.create(
    model="gpt-5.4",
    messages=[{"role": "user", "content": "What is 15 * 37? Think step by step."}],
)
msg = resp.choices[0].message
print("Reasoning:", getattr(msg, "reasoning_content", None))
print("Answer:", msg.content)

# Streaming (reasoning_content chunks arrive before content chunks)
for chunk in client.chat.completions.create(
    model="gpt-5.4",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
):
    delta = chunk.choices[0].delta
    # reasoning_content: thinking process (if present)
    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
        print(delta.reasoning_content, end="", flush=True)
    # content: final answer
    if delta.content:
        print(delta.content, end="", flush=True)

# Function calling
resp = client.chat.completions.create(
    model="gpt-5.4",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    }],
)
if resp.choices[0].finish_reason == "tool_calls":
    print(resp.choices[0].message.tool_calls)
```

## Reasoning Model Support

All Codex models are reasoning models. The server outputs the thinking process via the `reasoning_content` field, compatible with DeepSeek R1 / vLLM format.

### Non-streaming response

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "The answer is 555.",
      "reasoning_content": "I need to calculate 15 * 37. Let me break this down..."
    },
    "finish_reason": "stop"
  }]
}
```

### Streaming response

Reasoning chunks arrive first (with `delta.reasoning_content`), followed by content chunks (with `delta.content`):

```
data: {"choices":[{"delta":{"role":"assistant"}}]}
data: {"choices":[{"delta":{"reasoning_content":"I need to"}}]}
data: {"choices":[{"delta":{"reasoning_content":" calculate..."}}]}
data: {"choices":[{"delta":{"content":"The answer"}}]}
data: {"choices":[{"delta":{"content":" is 555."}}]}
data: {"choices":[{"delta":{},"finish_reason":"stop"}]}
data: [DONE]
```

### Reasoning effort control

Use `reasoning_effort` to control thinking depth:

| Value | Description |
|-------|-------------|
| `low` | Minimal thinking, fast responses |
| `medium` | Balanced (default) |
| `high` | Deep reasoning |

```bash
curl http://localhost:18888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-5.4","messages":[...],"reasoning_effort":"high"}'
```

## Parameter Compatibility

The server accepts all standard OpenAI / vLLM Chat Completions parameters. Supported parameters are mapped to the Codex Responses API format; unsupported ones are silently ignored (no errors).

### Supported parameters (forwarded to Codex)

| Parameter | Description |
|-----------|-------------|
| `model` | Model ID |
| `messages` | Conversation messages (system, user, assistant, tool) |
| `stream` | Enable streaming SSE |
| `tools` | Function/tool definitions |
| `tool_choice` | Tool selection strategy (`auto`, `none`, `required`, etc.) |
| `parallel_tool_calls` | Allow parallel function calls |
| `reasoning_effort` | Thinking depth: `low`, `medium`, `high` |
| `reasoning` | Advanced: `{"effort": "high", "summary": "detailed"}` |

### Silently ignored parameters (no error)

`temperature`, `top_p`, `top_k`, `max_tokens`, `max_completion_tokens`, `n`, `stop`, `presence_penalty`, `frequency_penalty`, `repetition_penalty`, `logit_bias`, `logprobs`, `top_logprobs`, `seed`, `response_format`, `user`, `stream_options`, `service_tier`, `min_p`, `best_of`, `suffix`, `guided_json`, `guided_regex`, `guided_choice`, `guided_grammar`, etc.

This means you can point any OpenAI SDK or vLLM client at this server without modifying request parameters.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/v1/models` | List available models |
| GET | `/v1/models/{id}` | Get model details |
| POST | `/v1/chat/completions` | Chat completion (streaming + non-streaming) |
| POST | `/v1/responses` | OpenAI Responses API proxy |
| GET | `/v1/logs` | Query API call logs |
| GET | `/v1/logs/stats` | Log statistics |
| GET | `/v1/logs/export` | Download logs as JSONL |
| POST | `/v1/logs/export/file` | Export logs to a local file |

### Log Query Parameters

```
GET /v1/logs?endpoint=chat/completions&model=gpt-5.4&limit=50&offset=0
```

### Export to File

```bash
curl http://localhost:18888/v1/logs/export/file \
  -H "Content-Type: application/json" \
  -d '{"filename": "my_export.jsonl"}'
```

Files are exported to `~/.codex/exports/` by default.

### SQLite Direct Access

```bash
sqlite3 -header -column ~/.codex/api_logs.db \
  "SELECT id, endpoint, model, status, total_tokens, duration_ms,
          datetime(created_at,'unixepoch','localtime') as time
   FROM api_logs ORDER BY id DESC LIMIT 10;"
```

## Configuration

All settings are configurable via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `CODEX_HOST` | `127.0.0.1` | Server bind address |
| `CODEX_PORT` | `18888` | Server port |
| `CODEX_API_KEY` | *(none)* | API key for client authentication (recommended) |
| `CODEX_AUTH_PATH` | `~/.codex/auth.json` | Path to Codex OAuth credentials |
| `CODEX_BASE_URL` | `https://chatgpt.com/backend-api` | Upstream Codex API base URL |
| `CODEX_CLIENT_ID` | *(built-in)* | OAuth client ID for token refresh |
| `CODEX_DB_PATH` | `~/.codex/api_logs.db` | SQLite database path |
| `CODEX_EXPORT_DIR` | `~/.codex/exports` | Allowed directory for file exports |
| `CODEX_MAX_BODY_BYTES` | `10485760` | Max request body size (10 MB) |

### Security Recommendations

- Set `CODEX_API_KEY` to restrict access to the server
- Keep the default `CODEX_HOST=127.0.0.1` for local-only access
- If exposing externally, use a reverse proxy with TLS

## Running Tests

```bash
# Start the server first, then:
bash test.sh
```

## License

MIT
