# Codex API Server

OpenAI API-compatible proxy server that uses ChatGPT Plus (Codex) OAuth credentials to provide standard `/v1/chat/completions` and `/v1/responses` endpoints.

## Features

- **OpenAI API compatible** — works with any OpenAI SDK or tool
- **Auto token refresh** — reads `~/.codex/auth.json`, refreshes OAuth tokens before expiry
- **Streaming SSE** — full Server-Sent Events support for chat completions
- **Responses API** — proxy passthrough for OpenAI's `/v1/responses` endpoint
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
```

### With OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:18888/v1", api_key="any")

# List models
models = client.models.list()

# Chat completion
resp = client.chat.completions.create(
    model="gpt-5.4",
    messages=[{"role": "user", "content": "Hello"}],
)
print(resp.choices[0].message.content)

# Streaming
for chunk in client.chat.completions.create(
    model="gpt-5.4",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
):
    print(chunk.choices[0].delta.content or "", end="")
```

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
