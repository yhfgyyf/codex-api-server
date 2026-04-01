import logging
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from auth import TokenManager
from config import CODEX_MODELS, EXPORT_DIR, LOCAL_API_KEY, MAX_BODY_BYTES, SERVER_HOST, SERVER_PORT
from db import close_db, count_logs, export_jsonl_file, export_jsonl_iter, init_db, query_logs
from models import get_model_info, get_model_list
from proxy import OpenAIProxy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("codex-api-server")

token_manager: TokenManager | None = None
proxy: OpenAIProxy | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global token_manager, proxy
    init_db()
    token_manager = TokenManager()
    proxy = OpenAIProxy(token_manager)
    if not LOCAL_API_KEY:
        logger.warning("No CODEX_API_KEY set — server accepts unauthenticated requests.")
    logger.info("Server started on %s:%d. Token valid: %s", SERVER_HOST, SERVER_PORT, token_manager.is_valid)
    yield
    if proxy:
        await proxy.close()
    close_db()


app = FastAPI(title="Codex API Server", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _check_api_key(request: Request):
    if not LOCAL_API_KEY:
        return
    auth_header = request.headers.get("authorization", "")
    key = auth_header[7:] if auth_header.startswith("Bearer ") else auth_header
    if key != LOCAL_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


async def _read_body(request: Request) -> dict:
    """Read and parse JSON body with size limit."""
    body_bytes = await request.body()
    if len(body_bytes) > MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail=f"Request body too large (limit {MAX_BODY_BYTES} bytes)")
    try:
        return __import__("json").loads(body_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")


# -- Health --

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "token_valid": token_manager.is_valid if token_manager else False,
        "models": CODEX_MODELS,
    }


# -- Models --

@app.get("/v1/models")
async def list_models(request: Request):
    _check_api_key(request)
    return get_model_list().model_dump()


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str, request: Request):
    _check_api_key(request)
    info = get_model_info(model_id)
    if not info:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    return info.model_dump()


# -- Chat Completions --

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    _check_api_key(request)
    body = await _read_body(request)
    stream = body.get("stream", False)

    if stream:
        return StreamingResponse(
            proxy.chat_completions_stream(body),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    result = await proxy.chat_completions(body)
    status = 200 if "error" not in result else (result.get("error", {}).get("status", 500))
    return JSONResponse(content=result, status_code=status if isinstance(status, int) else 500)


# -- Anthropic Messages API --

async def _handle_anthropic_messages(request: Request):
    _check_api_key(request)
    body = await _read_body(request)
    stream = body.get("stream", False)

    if stream:
        return StreamingResponse(
            proxy.anthropic_messages_stream(body),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    result = await proxy.anthropic_messages(body)
    status_code = 200 if result.get("type") != "error" else 400
    return JSONResponse(content=result, status_code=status_code)


app.post("/v1/messages")(_handle_anthropic_messages)
app.post("/messages")(_handle_anthropic_messages)
app.post("/v1/v1/messages")(_handle_anthropic_messages)


async def _handle_count_tokens(request: Request):
    """Anthropic count_tokens endpoint — return an estimate based on char count."""
    _check_api_key(request)
    body = await _read_body(request)
    messages = body.get("messages", [])
    system = body.get("system", "")

    # Rough estimate: ~4 chars per token for English, ~2 for CJK
    char_count = len(str(system))
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    char_count += len(block.get("text", "")) + len(block.get("thinking", ""))
        elif isinstance(content, str):
            char_count += len(content)
    estimated_tokens = max(1, char_count // 3)

    return JSONResponse(content={"input_tokens": estimated_tokens})


app.post("/v1/messages/count_tokens")(_handle_count_tokens)
app.post("/messages/count_tokens")(_handle_count_tokens)
app.post("/v1/v1/messages/count_tokens")(_handle_count_tokens)


# -- Responses API --

@app.post("/v1/responses")
async def responses(request: Request):
    _check_api_key(request)
    body = await _read_body(request)
    stream = body.get("stream", False)

    if stream:
        return StreamingResponse(
            proxy.responses_proxy_stream(body),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    result = await proxy.responses_proxy(body)
    return JSONResponse(content=result)


# -- Logs --

@app.get("/v1/logs")
async def get_logs(
    request: Request,
    endpoint: str | None = Query(None, description="Filter: chat/completions or responses"),
    model: str | None = Query(None, description="Filter by model name"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    _check_api_key(request)
    logs = await query_logs(endpoint=endpoint, model=model, limit=limit, offset=offset)
    total = await count_logs(endpoint=endpoint, model=model)
    return {
        "object": "list",
        "total": total,
        "limit": limit,
        "offset": offset,
        "data": logs,
    }


@app.get("/v1/logs/stats")
async def get_logs_stats(request: Request):
    _check_api_key(request)
    total = await count_logs()
    chat_count = await count_logs(endpoint="chat/completions")
    responses_count = await count_logs(endpoint="responses")
    return {
        "total": total,
        "chat_completions": chat_count,
        "responses": responses_count,
    }


# -- Export --

@app.get("/v1/logs/export")
async def export_logs_jsonl(
    request: Request,
    endpoint: str | None = Query(None),
    model: str | None = Query(None),
):
    """Download logs as JSONL."""
    _check_api_key(request)
    lines = await export_jsonl_iter(endpoint=endpoint, model=model)
    content = "\n".join(lines) + "\n" if lines else ""

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"codex_api_logs_{timestamp}.jsonl"

    return StreamingResponse(
        iter([content.encode("utf-8")]),
        media_type="application/jsonl",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/v1/logs/export/file")
async def export_logs_to_file(request: Request):
    """Export logs to a JSONL file within the allowed export directory."""
    _check_api_key(request)
    body = await _read_body(request)
    filename = body.get("filename")
    if not filename:
        raise HTTPException(status_code=400, detail="Missing 'filename' field")

    # Only accept a plain filename, not a path
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename — must be a plain name, not a path")

    filepath = str(EXPORT_DIR / filename)
    endpoint = body.get("endpoint")
    model = body.get("model")

    try:
        count, resolved = await export_jsonl_file(filepath, endpoint=endpoint, model=model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"status": "ok", "path": str(resolved), "records": count}


# -- Main --

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        log_level="info",
        access_log=True,
    )
