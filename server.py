import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from auth import TokenManager
from config import CODEX_MODELS, LOCAL_API_KEY, SERVER_HOST, SERVER_PORT
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
    logger.info(
        "Server started. Token valid: %s, expires: %s, account: %s",
        token_manager.is_valid,
        time.ctime(token_manager.expires_at),
        token_manager.account_id,
    )
    yield
    if proxy:
        await proxy.close()
    close_db()


app = FastAPI(title="Codex API Server", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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


# ── Health ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "token_valid": token_manager.is_valid if token_manager else False,
        "token_expires": time.ctime(token_manager.expires_at) if token_manager else None,
        "account_id": token_manager.account_id if token_manager else None,
        "models": CODEX_MODELS,
    }


# ── Models ───────────────────────────────────────────────────────────────

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


# ── Chat Completions ─────────────────────────────────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    _check_api_key(request)
    body = await request.json()
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


# ── Responses API ────────────────────────────────────────────────────────

@app.post("/v1/responses")
async def responses(request: Request):
    _check_api_key(request)
    body = await request.json()
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


# ── Logs Query ───────────────────────────────────────────────────────────

@app.get("/v1/logs")
async def get_logs(
    request: Request,
    endpoint: str | None = Query(None, description="Filter: chat/completions or responses"),
    model: str | None = Query(None, description="Filter by model name"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    _check_api_key(request)
    logs = query_logs(endpoint=endpoint, model=model, limit=limit, offset=offset)
    total = count_logs(endpoint=endpoint, model=model)
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
    total = count_logs()
    chat_count = count_logs(endpoint="chat/completions")
    responses_count = count_logs(endpoint="responses")
    return {
        "total": total,
        "chat_completions": chat_count,
        "responses": responses_count,
    }


# ── Export JSONL ─────────────────────────────────────────────────────────

@app.get("/v1/logs/export")
async def export_logs_jsonl(
    request: Request,
    endpoint: str | None = Query(None),
    model: str | None = Query(None),
):
    """Stream export as JSONL download."""
    _check_api_key(request)
    lines = export_jsonl_iter(endpoint=endpoint, model=model)
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
    """Export logs to a local JSONL file. Body: {"path": "/path/to/file.jsonl", "endpoint": null, "model": null}"""
    _check_api_key(request)
    body = await request.json()
    filepath = body.get("path")
    if not filepath:
        raise HTTPException(status_code=400, detail="Missing 'path' field")

    filepath = Path(filepath).expanduser()
    endpoint = body.get("endpoint")
    model = body.get("model")

    count = export_jsonl_file(filepath, endpoint=endpoint, model=model)
    return {"status": "ok", "path": str(filepath), "records": count}


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        log_level="info",
        access_log=True,
    )
