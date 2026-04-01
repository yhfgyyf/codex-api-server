# Codex API Server

基于 ChatGPT Plus (Codex) OAuth 凭据的 OpenAI API 兼容代理服务器，提供标准 `/v1/chat/completions`、`/v1/responses` 和 Anthropic `/v1/messages` 接口。

## 功能特性

- **OpenAI API 兼容** — 支持任何 OpenAI SDK、vLLM 客户端或工具直连
- **Anthropic Messages API** — 完整 `/v1/messages` 支持，含 `thinking` 推理内容块
- **推理模型支持** — 以 DeepSeek/vLLM 格式流式输出 `reasoning_content`（思考过程）
- **函数调用** — 完整的 tool/function calling 支持，含流式 tool_calls
- **自动刷新 Token** — 读取 `~/.codex/auth.json`，在过期前自动刷新 OAuth Token
- **流式 SSE** — 所有接口均支持 Server-Sent Events 流式输出
- **Responses API** — OpenAI `/v1/responses` 端点透传代理
- **vLLM 参数兼容** — 接受所有 OpenAI/vLLM 参数，不支持的参数静默忽略（不报错）
- **请求日志** — 所有 API 调用存储在 SQLite 数据库中，含完整请求/响应数据
- **JSONL 导出** — 下载或导出日志，用于分析和微调
- **异步高并发** — 基于 FastAPI + httpx 构建，连接池复用

## 支持的模型

- `gpt-5.4`
- `gpt-5.3-codex`
- `gpt-5.3-codex-spark`
- `gpt-5.2-codex`
- `gpt-5.1-codex`

## 前置条件

- Python 3.11+
- 有效的 `~/.codex/auth.json` 文件（ChatGPT OAuth 凭据，来自 [OpenAI Codex CLI](https://github.com/openai/codex)）

## 安装

```bash
# 创建 conda 环境
conda create -n codex-api-server python=3.11 -y
conda activate codex-api-server

# 安装依赖
pip install -r requirements.txt
```

## 快速开始

```bash
conda activate codex-api-server
python server.py
```

服务默认启动在 `http://127.0.0.1:18888`。

## 使用方法

### 使用 curl

```bash
# 列出模型
curl http://localhost:18888/v1/models

# 对话补全
curl http://localhost:18888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-5.4","messages":[{"role":"user","content":"你好"}]}'

# 流式对话补全
curl -N http://localhost:18888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-5.4","messages":[{"role":"user","content":"你好"}],"stream":true}'

# 控制推理深度
curl http://localhost:18888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-5.4","messages":[{"role":"user","content":"逐步计算 15*37"}],"reasoning_effort":"high"}'
```

### 使用 OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:18888/v1", api_key="any")

# 列出模型
models = client.models.list()

# 对话补全（含推理输出）
resp = client.chat.completions.create(
    model="gpt-5.4",
    messages=[{"role": "user", "content": "15 乘以 37 等于多少？请逐步推理。"}],
)
msg = resp.choices[0].message
print("推理过程:", getattr(msg, "reasoning_content", None))
print("回答:", msg.content)

# 流式输出（reasoning_content 在 content 之前到达）
for chunk in client.chat.completions.create(
    model="gpt-5.4",
    messages=[{"role": "user", "content": "你好"}],
    stream=True,
):
    delta = chunk.choices[0].delta
    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
        print(delta.reasoning_content, end="", flush=True)
    if delta.content:
        print(delta.content, end="", flush=True)

# 函数调用
resp = client.chat.completions.create(
    model="gpt-5.4",
    messages=[{"role": "user", "content": "东京现在天气怎么样？"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定地点的当前天气",
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

### 使用 Anthropic Python SDK

```python
import anthropic

# 注意：base_url 不要包含 /v1（SDK 会自动添加）
client = anthropic.Anthropic(base_url="http://localhost:18888", api_key="any")

# 基础消息
resp = client.messages.create(
    model="gpt-5.4",
    max_tokens=1024,
    messages=[{"role": "user", "content": "你好"}],
)
print(resp.content[0].text)

# 扩展思考（推理模型）
resp = client.messages.create(
    model="gpt-5.4",
    max_tokens=4096,
    thinking={"type": "enabled", "budget_tokens": 10000},
    messages=[{"role": "user", "content": "15 乘以 37 等于多少？请展示推理过程。"}],
)
for block in resp.content:
    if block.type == "thinking":
        print("思考过程:", block.thinking)
    elif block.type == "text":
        print("回答:", block.text)

# 流式输出
with client.messages.stream(
    model="gpt-5.4",
    max_tokens=1024,
    messages=[{"role": "user", "content": "从 1 数到 5"}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

### 使用 curl（Anthropic 格式）

```bash
# Anthropic Messages API
curl http://localhost:18888/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-5.4","max_tokens":1024,"messages":[{"role":"user","content":"你好"}]}'

# 启用思考过程
curl http://localhost:18888/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-5.4","max_tokens":4096,"thinking":{"type":"enabled","budget_tokens":10000},"messages":[{"role":"user","content":"计算 15*37"}]}'

# 流式输出
curl -N http://localhost:18888/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-5.4","max_tokens":1024,"stream":true,"messages":[{"role":"user","content":"你好"}]}'
```

## Anthropic Messages API

服务器实现了 Anthropic Messages API（`/v1/messages` 和 `/messages`），兼容任何 Anthropic SDK 或 vLLM Anthropic 端点客户端。

### 非流式响应

```json
{
  "id": "msg_abc123",
  "type": "message",
  "role": "assistant",
  "model": "gpt-5.4",
  "content": [
    {"type": "thinking", "thinking": "我需要计算..."},
    {"type": "text", "text": "答案是 555。"}
  ],
  "stop_reason": "end_turn",
  "usage": {"input_tokens": 29, "output_tokens": 70}
}
```

### 流式响应

使用 Anthropic SSE 格式（`event:` + `data:` 行）：

```
event: message_start
data: {"type":"message_start","message":{"id":"msg_abc","role":"assistant",...}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"我需要..."}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"答案是"}}

event: content_block_stop
data: {"type":"content_block_stop","index":1}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":70}}

event: message_stop
data: {"type":"message_stop"}
```

### 扩展思考

通过 `thinking` 参数启用思考过程。`budget_tokens` 值映射到 Codex 推理深度：

| budget_tokens | Codex 推理深度 |
|---------------|---------------|
| <= 2000 | `low`（低） |
| <= 8000 | `medium`（中） |
| > 8000 | `high`（高） |

### 支持的 Anthropic 参数

| 参数 | 说明 |
|------|------|
| `model` | 模型 ID |
| `messages` | 对话消息（user、assistant 角色，含内容块） |
| `system` | 系统提示词（字符串或内容块数组） |
| `max_tokens` | Anthropic SDK 必需参数（Codex 静默忽略） |
| `stream` | 启用流式 SSE |
| `thinking` | 扩展思考：`{"type": "enabled", "budget_tokens": N}` |
| `tools` | 工具定义（Anthropic 格式，含 `input_schema`） |
| `tool_choice` | 工具选择策略（`auto`、`any`、`tool`） |

其他 Anthropic 特有参数（`temperature`、`top_p`、`top_k`、`stop_sequences`、`metadata`）静默忽略。

## 推理模型支持

所有 Codex 模型都是推理模型。服务器通过 `reasoning_content` 字段输出思考过程，兼容 DeepSeek R1 / vLLM 格式。

### 非流式响应

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "答案是 555。",
      "reasoning_content": "我需要计算 15 * 37，分步骤来..."
    },
    "finish_reason": "stop"
  }]
}
```

### 流式响应

推理内容（`delta.reasoning_content`）先到达，随后是正式回答（`delta.content`）：

```
data: {"choices":[{"delta":{"role":"assistant"}}]}
data: {"choices":[{"delta":{"reasoning_content":"我需要"}}]}
data: {"choices":[{"delta":{"reasoning_content":"计算..."}}]}
data: {"choices":[{"delta":{"content":"答案是"}}]}
data: {"choices":[{"delta":{"content":" 555。"}}]}
data: {"choices":[{"delta":{},"finish_reason":"stop"}]}
data: [DONE]
```

### 推理深度控制

使用 `reasoning_effort` 控制思考深度：

| 值 | 说明 |
|----|------|
| `low` | 极简思考，快速响应 |
| `medium` | 平衡模式（默认） |
| `high` | 深度推理 |

```bash
curl http://localhost:18888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-5.4","messages":[...],"reasoning_effort":"high"}'
```

## 参数兼容性

服务器接受所有标准 OpenAI / vLLM Chat Completions 参数。支持的参数映射到 Codex Responses API 格式；不支持的参数静默忽略（不报错）。

### 支持的参数（转发至 Codex）

| 参数 | 说明 |
|------|------|
| `model` | 模型 ID |
| `messages` | 对话消息（system、user、assistant、tool） |
| `stream` | 启用流式 SSE |
| `tools` | 函数/工具定义 |
| `tool_choice` | 工具选择策略（`auto`、`none`、`required` 等） |
| `parallel_tool_calls` | 允许并行函数调用 |
| `reasoning_effort` | 思考深度：`low`、`medium`、`high` |
| `reasoning` | 高级用法：`{"effort": "high", "summary": "detailed"}` |

### 静默忽略的参数（不报错）

`temperature`、`top_p`、`top_k`、`max_tokens`、`max_completion_tokens`、`n`、`stop`、`presence_penalty`、`frequency_penalty`、`repetition_penalty`、`logit_bias`、`logprobs`、`top_logprobs`、`seed`、`response_format`、`user`、`stream_options`、`service_tier`、`min_p`、`best_of`、`suffix`、`guided_json`、`guided_regex`、`guided_choice`、`guided_grammar` 等。

这意味着你可以将任何 OpenAI SDK 或 vLLM 客户端直接指向本服务器，无需修改请求参数。

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| GET | `/v1/models` | 列出可用模型 |
| GET | `/v1/models/{id}` | 获取模型详情 |
| POST | `/v1/chat/completions` | 对话补全（流式 + 非流式） |
| POST | `/v1/messages` | Anthropic Messages API（流式 + 非流式） |
| POST | `/v1/responses` | OpenAI Responses API 代理 |
| GET | `/v1/logs` | 查询 API 调用日志 |
| GET | `/v1/logs/stats` | 日志统计 |
| GET | `/v1/logs/export` | 下载 JSONL 日志 |
| POST | `/v1/logs/export/file` | 导出日志到本地文件 |

### 日志查询参数

```
GET /v1/logs?endpoint=chat/completions&model=gpt-5.4&limit=50&offset=0
```

### 导出到文件

```bash
curl http://localhost:18888/v1/logs/export/file \
  -H "Content-Type: application/json" \
  -d '{"filename": "my_export.jsonl"}'
```

文件默认导出到 `~/.codex/exports/` 目录。

### SQLite 直接访问

```bash
sqlite3 -header -column ~/.codex/api_logs.db \
  "SELECT id, endpoint, model, status, total_tokens, duration_ms,
          datetime(created_at,'unixepoch','localtime') as time
   FROM api_logs ORDER BY id DESC LIMIT 10;"
```

## 配置

所有配置均可通过环境变量设置：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `CODEX_HOST` | `127.0.0.1` | 服务绑定地址 |
| `CODEX_PORT` | `18888` | 服务端口 |
| `CODEX_API_KEY` | *（无）* | 客户端认证 API Key（建议设置） |
| `CODEX_AUTH_PATH` | `~/.codex/auth.json` | Codex OAuth 凭据路径 |
| `CODEX_BASE_URL` | `https://chatgpt.com/backend-api` | 上游 Codex API 地址 |
| `CODEX_CLIENT_ID` | *（内置）* | OAuth Client ID |
| `CODEX_DB_PATH` | `~/.codex/api_logs.db` | SQLite 数据库路径 |
| `CODEX_EXPORT_DIR` | `~/.codex/exports` | 文件导出目录 |
| `CODEX_DEFAULT_MODEL` | `gpt-5.4` | 未识别模型名的默认回退模型 |
| `CODEX_MODEL_ALIASES` | *（无）* | 模型名映射（如 `claude-sonnet-4=gpt-5.4`） |
| `CODEX_MAX_BODY_BYTES` | `10485760` | 最大请求体大小（10 MB） |

### 模型名映射

当使用非 Codex 模型名时（例如 Claude Code 发送 `claude-sonnet-4-20250514`），服务器自动映射到默认 Codex 模型（`gpt-5.4`）。

可通过环境变量自定义映射：

```bash
# 修改默认回退模型
export CODEX_DEFAULT_MODEL=gpt-5.3-codex

# 添加显式模型别名（逗号分隔 key=value）
export CODEX_MODEL_ALIASES="claude-sonnet-4=gpt-5.4,claude-haiku=gpt-5.3-codex-spark"
```

### 安全建议

- 设置 `CODEX_API_KEY` 以限制服务器访问
- 保持默认 `CODEX_HOST=127.0.0.1` 仅本地访问
- 如需外网暴露，请使用反向代理 + TLS

## 运行测试

```bash
# 先启动服务，然后：
bash test.sh
```

## 许可证

MIT
